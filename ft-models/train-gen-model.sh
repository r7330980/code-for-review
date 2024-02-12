#!/bin/bash -x


WANDB_ENTITY=XXXXX \
WANDB_PROJECT=XXXXX \
deepspeed --num_gpus 4 --master_port 14568 \
    src/train_bash.py \
    --stage sft \
    --model_name_or_path stem-model-path \
    --run_name train-gen \
    --dataset dirty-mysplit-mix \
    --do_train \
    --do_eval \
    --train_on_prompt \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --val_size 0.01 \
    --cutoff_len 4096 \
    --template vanilla \
    --finetuning_type full \
    --preprocessing_num_workers 16 \
    --output_dir XXXXXX \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 2 \
    --lr_scheduler_type cosine \
    --warmup_steps 1000 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --deepspeed config/ds_config.json \
    --bf16 \
    --flash_attn 
    