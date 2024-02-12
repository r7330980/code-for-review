# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.sft.metric import ComputeMetrics
from llmtuner.tuner.sft.trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments
import os


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")
    print("Dataset length: %d" % len(dataset))
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left" # use left-padding in generation

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None, # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args)
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    if data_args.dbg_data:
        # get_rank_id
        rank_id = int(os.getenv('LOCAL_RANK', -1))
        BATCH_SIZE = 16
        NUM_DBG_BATCHES = 10
        dbg_fout = open("dbg-rank%d.txt"%rank_id, "w")
        print("Writing debug data to dbg-rank%d.txt"%rank_id)
        for i in range(NUM_DBG_BATCHES):
            related_data_entries = [dataset[t] for t in range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)]
            # print(related_data_entries)
            collated_batch = data_collator(related_data_entries)
            dbg_fout.write("Batch %d\n" % i)
            dbg_fout.write("===================\n")
            dbg_fout.write("Input IDs:\n")
            dbg_fout.write(str(collated_batch["input_ids"]) + "\n")
            for k, input_ids in enumerate(collated_batch["input_ids"]):                
                dbg_fout.write("----------------------Example %d----------------------\n" % k)
                dbg_fout.write("Detokenized:\n")
                dbg_fout.write(tokenizer.decode(input_ids) + "\n")
                dbg_fout.write("-------------------------------\n")
                dbg_fout.write("Labels:\n")
                dbg_fout.write(str(collated_batch["labels"][k]) + "\n")
                dbg_fout.write("-------------------------------\n")
                to_detok = [i for i in collated_batch["labels"][k] if i != -100]
                dbg_fout.write(tokenizer.decode(to_detok) + "\n")
                dbg_fout.write("Length of output: %d\n" % len(to_detok))
                dbg_fout.write("Length of input: %d\n" % len(input_ids))
                dbg_fout.write("-------------------------------\n")
        dbg_fout.close()
        exit()                                        


    # Training
    if training_args.do_train:
        print("Here")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
