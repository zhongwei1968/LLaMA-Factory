# test at H100 GPUs

export PRINT_SAMPLE=true

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1

pkill -f "python.*launcher.py" 

datasets=sg_public_legal_20250311,sg_public_judgment_20250311,sso_texts_20241205,redpajama_local,ultrachat_200k,super_glue_chat

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train \
    --stage sft \
    --do_train \
    --dataset ${datasets} \
    --model_name_or_path ../models/Qwen3.5-9B \
    --template qwen3_5 \
    --max_samples 100000 \
    --lora_rank 32 \
    --dataset_dir ../data \
    --finetuning_type lora \
    --lora_target all \
    --additional_target embed_tokens \
    --cutoff_len 80000 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --learning_rate 1e-05 \
    --num_train_epochs 2.0 \
    --bf16 \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --lr_scheduler_type cosine \
    --output_dir /opt/dlami/nvme/tmp \
    --overwrite_cache \
    --overwrite_output_dir \
    --flash_attn fa2 \
    --quantization_bit 4 \
    --deepspeed examples/deepspeed/ds_z2_config.json

