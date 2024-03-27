import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from constants import *
import datasets
from datasets import load_metric
import logging
import transformers
import sys
import yaml

import os

access_token = os.environ['HF_TOKEN']

logger = logging.getLogger(__name__)


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        truncation_side="right",
    )
    return tokenizer


def get_model_and_tokenizer(model_name, use_lora, inference=False, device_map="auto", task_type=SEQ_2_SEQ_LM):
    if inference:
        if use_lora:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
                device_map=device_map,
                token=access_token
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                token=access_token
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            token=access_token
        )

        if use_lora:
            peft_config = LoraConfig(
                task_type=task_type,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )

            model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        truncation_side="right",
    )
    return model, tokenizer


SRC_LANG, SRC_LANG_CODE, SRC_LANG_CODE_2 = 'French', "fr", "fra_Latn"
TRG_LANG, TRG_LANG_CODE, TRG_LANG_CODE_2 = 'English', "en", "eng_Latn"


def main(training_args, hparams):
    model_name = hparams["model"]
    model, tokenizer = get_model_and_tokenizer(model_name, hparams["lora"])

    def tokenize_function(examples):
        examples = examples['translation']
        tokenized_examples = tokenizer(examples[SRC_LANG_CODE], padding="max_length", truncation=True,
                                       return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[TRG_LANG_CODE], padding="max_length", truncation=True, return_tensors="pt")[
                "input_ids"]
        tokenized_examples["labels"] = labels
        return tokenized_examples

    datasets.dataset_dict()
    train_set = datasets.load_dataset("tatoeba", lang1=SRC_LANG_CODE, lang2=TRG_LANG_CODE)
    train_set = train_set.map(tokenize_function)

    dev_set_src = datasets.load_dataset("facebook/flores", "eng_Latn")['dev']
    dev_set_trg = datasets.load_dataset("facebook/flores", "fra_Latn")['dev']

    # Create a new dataset with 'source' and 'target' columns
    dev_set = dev_set_src.map(
        lambda examples: {'translations': {SRC_LANG_CODE: examples['sentence'],
                                           TRG_LANG_CODE: dev_set_trg['sentence'][examples['id']]}},
        remove_columns=['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink', 'sentence']
    )
    dev_set = dev_set.map(tokenize_function)

    bleu_metric = load_metric("bleu")

    def compute_metrics(pred):
        predictions = pred.predictions
        label_ids = pred.label_ids

        predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                       predictions]
        references = [[tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=False)] for l in
                      label_ids]

        bleu_score = bleu_metric.compute(predictions=predictions, references=references)
        return {"bleu": bleu_score["score"]}

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_set["train"],
        eval_dataset=dev_set,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    transformers.utils.logging.set_verbosity_info()
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    with open('lora.yaml') as hpf:
        hparams = yaml.load(hpf, Loader=yaml.FullLoader)
    print(hparams["training_args"])
    training_args = TrainingArguments(**hparams["training_args"])

    main(training_args, hparams)
