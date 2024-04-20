import time

import evaluate
import numpy as np
import pandas as pd
from adapters import AutoAdapterModel, AdapterTrainer
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
)

# Configuration and global settings
TASK = "sst2"
MODEL_CHECKPOINT = "distilbert-base-uncased"
BATCH_SIZE = 16
OUTPUT_DIR = "./training_output"
FAST_TESTING = False  # Toggle for quick testing or full training


def load_data(fast_testing=False):
    """Loads the dataset and metric based on the specified task, with an option for fast testing."""
    split = {'train': 'train[:500]', 'validation': 'validation[:100]'} if fast_testing else None
    dataset = load_dataset("glue", TASK, split=split or None)
    metric = evaluate.load("glue", TASK)
    return dataset, metric


def setup_tokenizer():
    """Initializes and returns the tokenizer from the pretrained model checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
    return tokenizer


def encode_data(tokenizer, dataset):
    """Encodes the dataset using the specified tokenizer and formats it for training."""
    def encode_batch(batch):
        return tokenizer(batch["sentence"], max_length=80, truncation=True, padding="max_length")

    dataset = dataset.map(encode_batch, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def get_training_arguments(model_type):
    """Returns tailored training arguments for adapter and LLM."""
    common_args = {
        "learning_rate": 1e-4 if model_type == "adapter" else 2e-5,
        "num_train_epochs": 1 if FAST_TESTING else 6,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "logging_steps": 500 if FAST_TESTING else 200,
        "evaluation_strategy": "no" if FAST_TESTING else "epoch",
        "save_strategy": "no" if FAST_TESTING else "epoch",
        "overwrite_output_dir": True,
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": 'accuracy',
        "output_dir": OUTPUT_DIR,
        "weight_decay": 0.01 if model_type == "llm" else 0  # Typically, no weight decay for adapters
    }
    return TrainingArguments(**common_args)


def train_model(model, dataset, tokenizer, model_type):
    """Generic function to train a model, used for both adapters and LLMs."""
    args = get_training_arguments(model_type)
    if model_type == "adapter":
        trainer = AdapterTrainer(model=model, args=args, train_dataset=dataset["train"],
                                 eval_dataset=dataset["validation"], compute_metrics=compute_metrics)
    else:
        trainer = Trainer(model=model, args=args, train_dataset=dataset["train"],
                          eval_dataset=dataset["validation"], tokenizer=tokenizer, compute_metrics=compute_metrics)

    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time
    print(f"{model_type.capitalize()} training completed in {elapsed_time:.2f} seconds.")

    results = trainer.evaluate()
    return results


def compute_metrics(eval_pred):
    """Compute accuracy metrics for evaluation predictions."""
    predictions, labels = eval_pred
    return {"accuracy": (np.argmax(predictions, axis=1) == labels).mean()}


def main(train_type):
    dataset, _ = load_data(fast_testing=FAST_TESTING)
    tokenizer = setup_tokenizer()
    dataset = encode_data(tokenizer, dataset)
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT, num_labels=2)

    if train_type == "adapter":
        model = AutoAdapterModel.from_pretrained(MODEL_CHECKPOINT, config=config)
        model.add_adapter(TASK, config="seq_bn")
        model.add_classification_head(TASK, num_labels=2, id2label={0: "👎", 1: "👍"})
        model.train_adapter(TASK)
    elif train_type == "llm":
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, config=config)
    else:
        raise ValueError("Invalid training type specified. Use 'adapter' or 'llm'.")

    results = train_model(model, dataset, tokenizer, train_type)

    # Save results to CSV
    pd.DataFrame([results]).to_csv(f"{OUTPUT_DIR}/{train_type}_results.csv", index=False)


if __name__ == "__main__":
    main("llm")  # 'adapter' or 'llm'
