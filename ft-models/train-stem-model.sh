#!/bin/bash -x


WANDB_ENTITY=XXXXX \
WANDB_PROJECT=XXXXX \
deepspeed --num_gpus 8 --master_port 12345 \
    src/train_bash.py \
    --stage sft \
    --model_name_or_path codellama/CodeLlama-7b-Instruct-hf \
    --run_name gennm-stem-model \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --train_on_prompt \
    --eval_steps 500 \
    --val_size 0.01 \
    --cutoff_len 4096 \
    --dataset dirty-mysplit \
    --template default \
    --finetuning_type full \
    --preprocessing_num_workers 16 \
    --output_dir XXXXX \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 2 \
    --lr_scheduler_type linear \
    --warmup_steps 2000 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --plot_loss \
    --deepspeed config/ds_config.json \
    --bf16 \
    --flash_attn
