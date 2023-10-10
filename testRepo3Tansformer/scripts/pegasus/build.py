
import sentencepiece as spm


spm.SentencePieceTrainer.train('--input= 1nTr1n$!cS3cur!ty --model_prefix=test_sentencepiece_no_bos --bos_id=-1 --unk_id=2  --eos_id=1  --vocab_size=1000')

