python translator.py \
    --source_path=data/en-zh/UNv1.0.en-zh.zh \
    --target_path=data/en-zh/UNv1.0.en-zh.en \
    --n_iters=100000 \
    --save_interval=10000 \
    --savedir=ckp/zh_en_finetune.100000iter/ \
    --encoder_name=bert-base-chinese \
    --decoder_name=bert-base-uncased \
    --lr=1e-3
