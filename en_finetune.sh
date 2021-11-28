python finetune.py \
    --source_path=data/en-zh/UNv1.0.en-zh.en \
    --n_iters=90001 \
    --save_interval=5000 \
    --log_interval=100 \
    --savedir=ckp/finetune.gpt2_wordemb_only.en/ \
    --lr=1e-3 \
    --pretrained_path=ckp/finetune.gpt2_wordemb_only.en/10000.pt \
    --pretrained_niters=10000 \
    --skiplines=1280000 \
    --batch_size=128 \
    --tune_wordemb_only

# --encoder_name=bert-base-uncased \
# --decoder_name=ckiplab/gpt2-base-chinese \
