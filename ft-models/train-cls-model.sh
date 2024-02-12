#!/bin/bash -x


WANDB_ENTITY=XXXXX \
WANDB_PROJECT=XXXXXX \
deepspeed --num_gpus 8 --master_port 14567 \
    src/train_bash.py \
    --stage lc \
    --model_name_or_path stem-model-path \
    --run_name train-cls-model \
    --class_id_to_str dirty-mysplit-intrain-id2str.json \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --train_on_prompt \
    --eval_steps 200 \
    --val_size 0.05 \
    --cutoff_len 4096 \
    --dataset dirty-mysplit-mix \
    --template vanilla \
    --finetuning_type full \
    --preprocessing_num_workers 16 \
    --output_dir XXXXXX \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 2 \
    --lr_scheduler_type cosine \
    --warmup_steps 1000 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --plot_loss \
    --deepspeed config/ds_config.json \
    --bf16 \
    --flash_attn