import time

import numpy as np
import pandas as pd
from adapters import AutoAdapterModel, AdapterTrainer
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
    DistilBertConfig
)

# Configuration and global settings
TASK = "sst2"
MODEL_CHECKPOINT = "distilbert-base-uncased"
BATCH_SIZE = 16
OUTPUT_DIR = "./training_output"


# def load_data():
#     """Loads the dataset and metric based on the specified task."""
#     dataset = load_dataset("glue", TASK)
#     metric = load_metric("glue", TASK)
#     return dataset, metric


def load_data():
    dataset = load_dataset("glue", TASK, split={'train': 'train[:500]', 'validation': 'validation[:100]'})
    metric = load_metric("glue", TASK)
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


def train_adapter(dataset):
    """Trains an adapter using the specified dataset."""
    config = DistilBertConfig.from_pretrained(MODEL_CHECKPOINT, num_labels=2)
    model = AutoAdapterModel.from_pretrained(MODEL_CHECKPOINT, config=config)
    model.add_adapter(TASK, config="seq_bn")
    model.add_classification_head(TASK, num_labels=2, id2label={0: "👎", 1: "👍"})
    model.train_adapter(TASK)

    training_args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=500,
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        evaluation_strategy="no",
        save_strategy="no",
    )

    # training_args = TrainingArguments(
    #     learning_rate=1e-4,
    #     num_train_epochs=6,
    #     per_device_train_batch_size=32,
    #     per_device_eval_batch_size=32,
    #     logging_steps=200,
    #     output_dir=OUTPUT_DIR,
    #     overwrite_output_dir=True,
    #     remove_unused_columns=False
    # )

    def compute_accuracy(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": (preds == p.label_ids).mean()}

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_accuracy
    )

    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time
    print(f"Adapter training completed in {elapsed_time:.2f} seconds.")

    results = trainer.evaluate()
    return results


def train_llm(dataset, tokenizer):
    """Trains a language model using the specified dataset and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)

    training_args = TrainingArguments(
        evaluation_strategy="no",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        output_dir=OUTPUT_DIR,
        logging_steps=500,
    )

    # training_args = TrainingArguments(
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=BATCH_SIZE,
    #     per_device_eval_batch_size=BATCH_SIZE,
    #     num_train_epochs=6,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     output_dir=OUTPUT_DIR
    # )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": (predictions == labels).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time
    print(f"LLM training completed in {elapsed_time:.2f} seconds.")

    results = trainer.evaluate()
    return results


def main(train_type):
    """Main function to initiate training based on the specified type ('adapter' or 'llm')."""
    dataset, _ = load_data()
    tokenizer = setup_tokenizer()
    dataset = encode_data(tokenizer, dataset)

    if train_type == "adapter":
        results = train_adapter(dataset)
    elif train_type == "llm":
        results = train_llm(dataset, tokenizer)
    else:
        raise ValueError("Invalid training type specified. Use 'adapter' or 'llm'.")

    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{OUTPUT_DIR}/{train_type}_results.csv", index=False)


if __name__ == "__main__":
    # main("adapter")
    main("llm")
