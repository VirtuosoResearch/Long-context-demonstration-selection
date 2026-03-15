python mamba_inference.py \
  --dataset sst2,cr,coin_flip \
  --run_mode train_eval \
  --train_split dev \
  --retrieval_split dev \
  --eval_split dev \
  --save_sidecar_path sidecar_sst2_cr_coinflip.pt