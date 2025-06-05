cd models
gunzip GoogleNews-vectors-negative300.bin.gz
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
rm crawl-300d-2M.vec.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip
rm crawl-300d-2M-subword.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz
gunzip cc.ja.300.vec.gz

Download Russian models
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz
gunzip cc.ru.300.vec.gz

Download RusVectores models
wget -O "geowac_tokens_none_fasttextskipgram_300_5_2020.zip" https://vectors.nlpl.eu/repository/20/214.zip
unzip geowac_tokens_none_fasttextskipgram_300_5_2020.zip -d geowac_tokens_none_fasttextskipgram_300_5_2020
rm geowac_tokens_none_fasttextskipgram_300_5_2020.zip

cd ..
python src/convert_model_to_torch.py