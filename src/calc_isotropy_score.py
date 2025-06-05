import json
import re
import math
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import sentence_transformers
import torch
from fire import Fire
from mteb import MTEB
from nltk import word_tokenize
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, WordEmbeddings
from torchtyping import TensorType as TT
from tqdm import tqdm

from all_but_the_top import AllButTheTop
from eval_utils import (
    UnigramProb,
    WrappedTokenizer,
    load_unigram_prob_enwiki_vocab_min200,
    load_word2vec_model,
    load_fasttext_model,
    remove_unused_words,
    load_unigram_prob_jawiki_top_20k,
    load_unigram_prob_ruvocab
)
from IsoScore import IsoScore
from SIF import SIF
from zipfian_whitening import UniformWhitening, ZipfianWhitening


TRANSFORM_CONFIG = {
    "normal": {
        "whitening_transformer_class": None,
        "pooling": [
            "mean",  # no whitening. normal mean pooling.
        ],},
    "uniform_pca": {
        "whitening_transformer_class": UniformWhitening,
        "pooling": [
            "centering_only",
            "whitening",
        ],},
    # "uniform_zca": {
    #     "whitening_transformer_class": UniformWhitening,
    #     "pooling": [
    #         "centering_only",
    #         "whitening",
    #     ],},
    # "uniform_cholesky": {
    #     "whitening_transformer_class": UniformWhitening,
    #     "pooling": [
    #         "centering_only",
    #         "whitening",
    #     ],},
    "zipfian_pca": {
        "whitening_transformer_class": ZipfianWhitening,
        "pooling": [
            "centering_only",
            "whitening",
        ],},
    # "zipfian_zca": {
    #     "whitening_transformer_class": ZipfianWhitening,
    #     "pooling": [
    #         "centering_only",
    #         "whitening",
    #     ],},
    # "zipfian_cholesky": {
    #     "whitening_transformer_class": ZipfianWhitening,
    #     "pooling": [
    #         "centering_only",
    #         "whitening",
    #     ],},
    "abtp": {
        "whitening_transformer_class": AllButTheTop,
        "pooling": ["component_removal"],
    }
}
# Downloaded from: https://github.com/kawine/usif/raw/71ffef5b6d7295c36354136bfc6728a10bd25d32/enwiki_vocab_min200.txt
PATH_ENWIKI_VOCAB_MIN200 = "data/enwiki_vocab_min200/enwiki vocab min200.txt"
PATH_RUVOCAB = "data/ruwiki_vocab/ruvocab.txt"

MODEL_NAME_TO_FREQ_FUNC = {
    "models/GoogleNews-vectors-negative300-torch": load_unigram_prob_enwiki_vocab_min200,
    "models/fasttext-ja-torch": load_unigram_prob_jawiki_top_20k,
    "models/fasttext-en-torch": load_unigram_prob_enwiki_vocab_min200,
    "models/fasttext-en-subword-torch": load_unigram_prob_enwiki_vocab_min200,
    "sentence-transformers/average_word_embeddings_glove.840B.300d": load_unigram_prob_enwiki_vocab_min200,
    "models/fasttext-ru-torch": load_unigram_prob_ruvocab,
    "models/geowac_tokens_none_fasttextskipgram_300_5_2020-torch": load_unigram_prob_ruvocab
}

def cos_sim(v1, v2):
    return torch.dot(v1, v2) / (torch.linalg.norm(v1) * torch.linalg.norm(v2))


def calc_zipfian_isotropy_score(
    W: TT["num_words", "hidden_dim"], p: TT["num_words"]
) -> Tuple[float, float]:
    # Measure the degree of centrality (the 1st moment of symmetry)
    mu: TT["hidden_dim"] = p @ W
    l: TT["hidden_dim"] = p @ torch.linalg.norm(W, dim=1)

    sym1 = torch.linalg.norm(mu) / l

    # Measure the degree of isotropy (the 2nd moment of symmetry)
    W_centered: TT["num_words", "hidden_dim"] = W - mu
    Wp = W_centered * torch.sqrt(p[:, None])
    U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)
    lambda_ = S**2
    d = torch.tensor(W.shape[1])
    sym2 = (
        (lambda_ / lambda_.sum())
        @ (torch.log(lambda_ / lambda_.sum()))
        / (-torch.log(d))
    )
    sym1 = 1 - sym1  # to match the range of the sym2

    return sym1, sym2


def calc_uniform_isotropy_score(
    W: TT["num_words", "hidden_dim"],
) -> Tuple[float, float]:
    # Measure the degree of centrality (the 1st moment of symmetry)
    mu: TT["hidden_dim"] = torch.mean(W, dim=0)
    l: TT["hidden_dim"] = torch.mean(torch.linalg.norm(W, dim=1))

    sym1 = torch.linalg.norm(mu) / l

    # Measure the degree of isotropy (the 2nd moment of symmetry)
    W_centered: TT["num_words", "hidden_dim"] = W - mu
    U, S, Vt = torch.linalg.svd(W_centered, full_matrices=False)
    lambda_ = S**2
    d = torch.tensor(W.shape[1])
    sym2 = (
        (lambda_ / lambda_.sum())
        @ (torch.log(lambda_ / lambda_.sum()))
        / (-torch.log(d))
    )
    sym1 = 1 - sym1  # to match the range of the sym2

    return sym1, sym2


def cos_sim(v1, v2):
    return torch.dot(v1, v2) / (torch.linalg.norm(v1) * torch.linalg.norm(v2))


# used running mean insted of concatination
def calc_cosine_score(W: torch.Tensor, n: int = 16) -> float:
    # XXX: If CUDA OOM error occurs, enlarge the n value.

    # Normalize the vectors
    W_norm = W / W.norm(dim=1, keepdim=True)

    # Split the tensor into n parts
    splits = torch.split(W_norm, W_norm.shape[0] // n + 1)

    cos_sim_mean = 0
    counter = 1

    # Compute cosine similarities within and between splits
    for i in tqdm(range(len(splits))):
        for j in range(i, len(splits)):
            cos_sim_ij = torch.mm(splits[i], splits[j].T)
            if i == j:
                triu_indices = torch.triu_indices(
                    cos_sim_ij.shape[0], cos_sim_ij.shape[1], offset=1
                )
                cos_sim_mean = cos_sim_mean * (counter-1)/counter + cos_sim_ij[triu_indices[0], triu_indices[1]].mean().item()/counter
                counter += 1
            else:
                cos_sim_mean = cos_sim_mean * (counter-1)/counter + cos_sim_ij.mean().item()/counter
                counter += 1

    return 1 - cos_sim_mean


def evaluate_isotropy_scores(
    model: SentenceTransformer,
    model_name: str,
    whitening_transformer: Union[UniformWhitening, ZipfianWhitening],
    whitening_name: str,
    pooling_mode: str,
    unigramprob_tensor: TT["num_words"],
    embedding_for_whitening: TT["num_words", "hidden_dim"],
    embedding_layer_index: int = 0,
    pooling_layer_index: int = 1,
    unused_vocab_ids: set[int] = None,
    task_name: Optional[str] = None,  # only for sif
) -> None:
    model_name = Path(model_name).name  # remove "SentenceTransformer/" prefix
    embedding_matrix = embedding_for_whitening  # Only consider the embeddings of the words in the frequency list.
    if whitening_transformer is None:
        # whitening_name = "normal"
        pass
    elif isinstance(whitening_transformer, ZipfianWhitening):
        # whitening_name = "zipfian_whitening"
        if pooling_mode == "centering_only":
            embedding_matrix = embedding_matrix - whitening_transformer.mu
        elif pooling_mode == "whitening":
            embedding_matrix = embedding_matrix - whitening_transformer.mu
            embedding_matrix = (
                embedding_matrix @ whitening_transformer.transformation_matrix
            )
        else:
            raise NotImplementedError(
                'Only "centering_only" and "whitening" pooling modes are supported for ZipfianWhitening.'
            )
    elif isinstance(whitening_transformer, UniformWhitening):
        # whitening_name = "uniform_whitening"
        if pooling_mode == "centering_only":
            embedding_matrix = embedding_matrix - whitening_transformer.mu
        elif pooling_mode == "whitening":
            embedding_matrix = embedding_matrix - whitening_transformer.mu
            embedding_matrix = (
                embedding_matrix @ whitening_transformer.transformation_matrix
            )
        else:
            raise NotImplementedError(
                'Only "centering_only" and "whitening" pooling modes are supported for UniformWhitening.'
            )
    elif isinstance(whitening_transformer, AllButTheTop):
        # whitening_name = "abtp"
        if pooling_mode == "component_removal":
            embedding_matrix = embedding_matrix - whitening_transformer.mu
            embedding_matrix = (
                embedding_matrix
                - (embedding_matrix @ whitening_transformer.common_components.T)
                @ whitening_transformer.common_components
            )
        else:
            raise NotImplementedError(
                'Only "component_removal" pooling mode is supported for AllButTheTop.'
            )
    elif isinstance(whitening_transformer, SIF):
        # whitening_name = "sif"
        if pooling_mode == "sif_w_component_removal":
            embedding_matrix = (
                model[embedding_layer_index].emb_layer.weight
            )  # Here use the original embeddings. This does not affect the result.
            embedding_matrix = (
                embedding_matrix
                - (embedding_matrix @ whitening_transformer.common_components.T)
                @ whitening_transformer.common_components
            )
            # Scaling by sif weights
            # SIF weights for unused words can be 0 (or p(w) = 0), since it is removed later or never used (blocked by the tokenizer).
            embedding_matrix = (
                embedding_matrix * whitening_transformer.sif_weights[:, None]
            )
            # Remove the unused embeddings, since the vocab size of the model is too big for cosine similarity computation.
            embedding_matrix, unigramprob_tensor = remove_unused_words(
                unused_vocab_ids, embedding_matrix, unigramprob_tensor
            )

        else:
            raise NotImplementedError(
                'Only "sif_w_component_removal" pooling mode is supported for SIF.'
            )
    else:
        raise NotImplementedError(
            'Only "ZipfianWhitening" and "UniformWhitening" and "AllButTheTop" and "SIF" are supported.'
        )
    print(
        f"Start: Model: {model_name}, Whitening {whitening_name}, Pooling: {pooling_mode}"
    )
    # save_dir name rule:
    # {model_name}/{task_name}/{whitening_transformer_name (e.g., zipfian_whitening, ...)}/{pooling_mode}
    if task_name is not None:
        pooling_mode += f"_{task_name}"
    save_dir_name = (
        f"results/{model_name}/isotropy_scores/{whitening_name}/{pooling_mode}"
    )

    sim1, sim2 = calc_zipfian_isotropy_score(embedding_matrix, unigramprob_tensor)
    sim1_uniform, sim2_uniform = calc_uniform_isotropy_score(embedding_matrix)

    cos_score = calc_cosine_score(embedding_matrix)
    iso_score = IsoScore(embedding_matrix)
    # save the results as json
    save_dir = Path(save_dir_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    # if the json file already exists, append the new results to the existing file.
    results = {
        "model_name": model_name,
        "whitening_name": whitening_name,
        "pooling_mode": pooling_mode,
        "sym1": sim1.item(),
        "sym2": sim2.item(),
        "sym1_uniform": sim1_uniform.item(),
        "sym2_uniform": sim2_uniform.item(),
        "cosine": cos_score,
        "iso_score": iso_score,
    }
    with open(save_dir / "isotropy_scores.json", "w") as f:
        json.dump(results, f)
    torch.save(embedding_matrix, f"{save_dir}/emb_matrix.pt", )

    pprint(results)


def main(model_name: str, topk: Optional[int] = None) -> None:
    embedding_layer_index = 0
    pooling_layer_index = 1
    if model_name == "models/GoogleNews-vectors-negative300":
        model: SentenceTransformer = load_word2vec_model(
            model_name, from_text_file=True
        )
    elif (
        model_name == "models/GoogleNews-vectors-negative300-torch"
        or model_name == "models/fasttext-ja-torch"
        or model_name == "models/fasttext-en-torch"
        or model_name == "models/fasttext-en-subword-torch"
        or model_name == "models/fasttext-ru-torch"
        or model_name == "models/geowac_tokens_none_fasttextskipgram_300_5_2020-torch"
    ):
        model: SentenceTransformer = load_word2vec_model(
            model_name, from_text_file=False
        )
    elif model_name == "models/crawl-300d-2M.vec":
        model: SentenceTransformer = load_fasttext_model(model_name)
    else:
        model = SentenceTransformer(model_name)

    # To match the experiment setting to the original paper
    model.tokenizer.stop_words = {}
    model.tokenizer.do_lower_case = True
    model.tokenizer = WrappedTokenizer(model.tokenizer)
    model_vocab_size = model[embedding_layer_index].emb_layer.weight.shape[0]
    unigramprob: UnigramProb = MODEL_NAME_TO_FREQ_FUNC[model_name](
        model.tokenizer, model_vocab_size, topk=topk
    )
    unigramprob_tensor: TT["num_words"] = unigramprob.prob.to(model.device)
    unsued_vocab_ids: set[int] = unigramprob.unused_vocab_ids
    model.tokenizer.original_tokenizer.stop_words = {
        model.tokenizer.vocab[index] for index in unsued_vocab_ids
    }

    # To reduce the noise for the whitening, remove the unused words from the embeddings and unigram probabilities.
    embedding_for_whitening, unigramprob_tensor = remove_unused_words(
        unsued_vocab_ids,
        model[embedding_layer_index].emb_layer.weight,
        unigramprob_tensor,
    )

    params = {
        "model": model,
        "model_name": model_name,
        "whitening_transformer": None,
        "embedding_layer_index": embedding_layer_index,
        "pooling_layer_index": pooling_layer_index,
        "embedding_for_whitening": embedding_for_whitening,
        "unigramprob_tensor": unigramprob_tensor,
    }
    # for non-sif methods
    for transform_name in TRANSFORM_CONFIG:
        # set as default
        whitening_mode = None

        params["pooling_mode"]: List[str] = TRANSFORM_CONFIG[transform_name]["pooling"]
        whitening_transformer = TRANSFORM_CONFIG[transform_name][
            "whitening_transformer_class"
        ]
        if whitening_transformer is not None:
            if issubclass(whitening_transformer, UniformWhitening) or issubclass(whitening_transformer, ZipfianWhitening):
                _, whitening_mode = transform_name.rsplit("_", 1)

        whitening_transformer = (
            None
            if whitening_transformer is None
            else whitening_transformer(mode=whitening_mode).fit(
                embedding_for_whitening, p=unigramprob_tensor
            )
        )
        params["whitening_transformer"] = whitening_transformer
        params["whitening_name"] = transform_name
        for pooling_mode in params["pooling_mode"]:
            params["pooling_mode"] = pooling_mode
            evaluate_isotropy_scores(**params)

    # SIF
    unigramprob: UnigramProb = MODEL_NAME_TO_FREQ_FUNC[model_name](
        model.tokenizer, model_vocab_size, topk=topk
    )
    unsued_vocab_ids: set[int] = unigramprob.unused_vocab_ids
    unigramprob_tensor: TT["num_words"] = unigramprob.prob.to(model.device)

    unigramprob_tensor[list(unsued_vocab_ids)] = 0
    unigramprob_tensor = unigramprob_tensor / unigramprob_tensor.sum()
    assert torch.allclose(unigramprob_tensor.sum(), torch.tensor(1.0))
    model.tokenizer.original_tokenizer.stop_words = {
        model.tokenizer.vocab[index] for index in unsued_vocab_ids
    }
    pooling_mode = "sif_w_component_removal"
    for task_name in ["SICK-R", "STSBenchmark"]:
        sif = SIF(model, task_name=task_name)
        sif.fit(None, unigramprob_tensor)
        evaluate_isotropy_scores(
            model=model,
            model_name=model_name,
            whitening_transformer=sif,
            whitening_name="sif",
            pooling_mode=pooling_mode,
            embedding_layer_index=embedding_layer_index,
            pooling_layer_index=pooling_layer_index,
            unigramprob_tensor=unigramprob_tensor,
            embedding_for_whitening=embedding_for_whitening,
            unused_vocab_ids=unsued_vocab_ids,
            task_name=task_name,
        )


if __name__ == "__main__":
    Fire(main)
