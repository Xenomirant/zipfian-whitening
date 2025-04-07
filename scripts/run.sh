# # preparation: calculate the norm of zipf/unif whitened vectors
# echo "Calculating norm for uniform processing"
# python src/calculate_uniform_norm.py --model_name models/GoogleNews-vectors-negative300-torch --whitening_mode "PCA"
# python src/calculate_uniform_norm.py --model_name models/GoogleNews-vectors-negative300-torch --whitening_mode "ZCA"
# python src/calculate_uniform_norm.py --model_name models/GoogleNews-vectors-negative300-torch --whitening_mode "Cholesky"
# python src/calculate_uniform_norm.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d --whitening_mode "PCA"
# python src/calculate_uniform_norm.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d --whitening_mode "ZCA"
# python src/calculate_uniform_norm.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d --whitening_mode "Cholesky"
# python src/calculate_uniform_norm.py --model_name models/fasttext-en-torch --whitening_mode "PCA"
# python src/calculate_uniform_norm.py --model_name models/fasttext-en-torch --whitening_mode "ZCA"
# python src/calculate_uniform_norm.py --model_name models/fasttext-en-torch --whitening_mode "Cholesky"
# python src/calculate_uniform_norm.py --model_name models/fasttext-en-subword-torch --whitening_mode "PCA"
# python src/calculate_uniform_norm.py --model_name models/fasttext-en-subword-torch --whitening_mode "ZCA"
# python src/calculate_uniform_norm.py --model_name models/fasttext-en-subword-torch --whitening_mode "Cholesky"
#
# echo "Calculating norm for zipfian processing"
# python src/calculate_zipfian_norm.py --model_name models/GoogleNews-vectors-negative300-torch --whitening_mode "PCA"
# python src/calculate_zipfian_norm.py --model_name models/GoogleNews-vectors-negative300-torch --whitening_mode "ZCA"
# python src/calculate_zipfian_norm.py --model_name models/GoogleNews-vectors-negative300-torch --whitening_mode "Cholesky"
# python src/calculate_zipfian_norm.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d --whitening_mode "PCA"
# python src/calculate_zipfian_norm.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d --whitening_mode "ZCA"
# python src/calculate_zipfian_norm.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d --whitening_mode "Cholesky"
# python src/calculate_zipfian_norm.py --model_name models/fasttext-en-torch --whitening_mode "PCA"
# python src/calculate_zipfian_norm.py --model_name models/fasttext-en-torch --whitening_mode "ZCA"
# python src/calculate_zipfian_norm.py --model_name models/fasttext-en-torch --whitening_mode "Cholesky"
# python src/calculate_zipfian_norm.py --model_name models/fasttext-en-subword-torch --whitening_mode "PCA"
# python src/calculate_zipfian_norm.py --model_name models/fasttext-en-subword-torch --whitening_mode "ZCA"
# python src/calculate_zipfian_norm.py --model_name models/fasttext-en-subword-torch --whitening_mode "Cholesky"

# run sts evaluation via MTEB
echo "Running evaluation on models"
## w/ wiki freq
python src/run_eval.py --model_name models/GoogleNews-vectors-negative300-torch
python src/run_eval.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d
python src/run_eval.py --model_name models/fasttext-en-torch
python src/run_eval.py --model_name models/fasttext-en-subword-torch
# python src/run_eval.py --model_name models/fasttext-ja-torch --task_names "[JSTS]"
## w/ test set freq
python src/run_eval.py --model_name models/GoogleNews-vectors-negative300-torch --in_batch True
python src/run_eval.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d --in_batch True
python src/run_eval.py --model_name models/fasttext-en-torch --in_batch True
python src/run_eval.py --model_name models/fasttext-en-subword-torch --in_batch True
# python src/run_eval.py --model_name models/fasttext-ja-torch --task_names "[JSTS]" --in_batch True

# calculate isotropy scores
echo "Calculating isotropy scores of various kinds"
python src/calc_isotropy_score.py --model_name models/GoogleNews-vectors-negative300-torch
python src/calc_isotropy_score.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d
python src/calc_isotropy_score.py --model_name models/fasttext-en-torch
python src/calc_isotropy_score.py --model_name models/fasttext-en-subword-torch

# run norm experiments
echo "Running norm experiments on average_word_embeddings_glove.840B.300d"
python src/run_norm_experiments.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d --topk 500
echo "Running norm experiments on GoogleNews-vectors-negative300-torch"
python src/run_norm_experiments.py --model_name models/GoogleNews-vectors-negative300-torch --topk 500
