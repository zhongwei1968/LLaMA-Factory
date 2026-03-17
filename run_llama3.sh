# test at H100 GPUs
pkill -f '/app/src/llamafactory/launcher.py'

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train \
    --stage sft \
    --do_train \
    --dataset qca_63_examples_sft_md_250822 \
    --model_name_or_path /models/Meta-Llama-3.1-8B-Instruct \
    --template llama3 \
    --max_samples 100000 \
    --lora_rank 32 \
    --dataset_dir /data \
    --finetuning_type lora \
    --lora_target all \
    --additional_target embed_tokens \
    --cutoff_len 24576 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --logging_steps 2 \
    --warmup_steps 5 \
    --save_steps 10 \
    --learning_rate 1e-05 \
    --num_train_epochs 2.0 \
    --fp16 \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 10 \
    --lr_scheduler_type cosine \
    --output_dir /models/checkpoints \
    --overwrite_cache \
    --overwrite_output_dir \
    --flash_attn fa2 \
    --quantization_bit 4
