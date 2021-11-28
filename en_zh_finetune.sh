python translator.py \
    --source_path=data/en-zh/UNv1.0.en-zh.en \
    --target_path=data/en-zh/UNv1.0.en-zh.zh \
    --n_iters=10001 \
    --save_interval=1000 \
    --log_interval=100 \
    --savedir=ckp/en_zh_finetune.after_en_finetune.wordemb_only/ \
    --encoder_name=ckp/finetune.gpt2_wordemb_only.en/100000/ \
    --decoder_name=uer/gpt2-chinese-cluecorpussmall \
    --lr=1e-3 \
    --batch_size=64 \
    --tune_wordemb_only \
    --debug

# --encoder_name=bert-base-uncased \
# --decoder_name=ckiplab/gpt2-base-chinese \
