python translator.py \
    --source_path=data/en-zh/UNv1.0.en-zh.en \
    --target_path=data/en-zh/UNv1.0.en-zh.zh \
    --n_iters=5000 \
    --save_interval=1000 \
    --savedir=ckp/en_zh_finetune/ \
    --encoder_name=bert-base-uncased \
    --decoder_name=bert-base-chinese \
    --lr=1e-2 \
    --batch_size=16 \
    --debug