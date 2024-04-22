# Standard library imports
import os
import time
import platform
from datetime import datetime

# Third-party imports for data handling and mathematics
import torch
import pandas as pd
import numpy as np

# Machine learning and metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# PyTorch utility for data handling
from torch.utils.data import Dataset

# Transformers library imports
import adapters
from adapters import LoRAConfig
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    AutoTokenizer,
    set_seed
)


""" # Load the training and evaluation data

# List of [model_name, adapter(bool), column_name] combinations
combinations = [
    ['bert-base-uncased', True, 'Analyst Note', 'adapter_config'],
    # Add more combinations as needed
]

# Load the training and evaluation data
train_df = pd.read_csv('1. data/final/train.csv')
eval_df = pd.read_csv('1. data/final/eval.csv')

for model_name, adapter, column_name in combinations:
    # Create an instance of CustomTransformerModel
    model = CustomTransformerModel(model_name=model_name, adapter=adapter, column_name=column_name)

    # Build the model
    model.build_model()

    # Train the model
    model.train(train_df, eval_df, epochs=3, batch_size=32, learning_rate=2e-5, seed=42, save_model=False)

"""


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class CustomTransformerModel:

    def __init__(self, model_name: str, adapter: bool = False, column_name: str = 'text', adapter_config=LoRAConfig()):
        self.model_name = model_name
        self.adapter = adapter
        self.column_name = column_name
        self.model = None
        self.tokenizer = None
        self.adapter_name = f"{self.model_name}_adapter"
        self.adapter_config = adapter_config


    def build_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.adapter:
            adapters.init(self.model)
            # Load and activate the adapter
            # Create an adapter configuration
            self.model.add_adapter(self.adapter_name, 
            config=self.adapter_config,
            set_active=True)
            self.model.train_adapter(self.adapter_name)

    def compute_metrics(self, p: EvalPrediction):
        predictions = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy_score(p.label_ids, predictions),
            "precision": precision_score(p.label_ids, predictions, average='weighted'),
            "recall": recall_score(p.label_ids, predictions, average='weighted'),
            "f1": f1_score(p.label_ids, predictions, average='weighted')
        }


    def train(self, df: pd.DataFrame, eval_df: pd.DataFrame, test_df: pd.DataFrame, epochs: int = 3, batch_size: int = 32, learning_rate: float = 2e-5, seed: int = 42, save_model: bool = False):
        set_seed(seed)
        if self.model is None or self.tokenizer is None:
            raise Exception("Model is not built. Call build_model first.")

        # Assume that the DataFrame has a 'label' column for labels
        train_dataset = TextDataset(df[self.column_name], df['Label'], self.tokenizer)
        dev_dataset = TextDataset(eval_df[self.column_name], eval_df['Label'], self.tokenizer)
        test_dataset = TextDataset(test_df[self.column_name], test_df['Label'], self.tokenizer)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_safetensors=False,             
            logging_dir='./logs',
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            logging_steps=200,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=self.compute_metrics
        )

        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time

        start_time = time.time()
        eval_metrics = trainer.evaluate()
        evaluation_time = time.time() - start_time

        # Get the number of model parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Get the hardware information
        hardware_info = platform.uname()
        
        date = datetime.now()


        # Create a DataFrame and save it as a CSV file
        data = {
            'model_name': [self.model_name],
            'adapter_name': [self.adapter_name],
            'adapter_config': [self.adapter_config],
            'column_name': [self.column_name],
            'training_data_length': [len(train_dataset)],
            'test_data_length': [len(dev_dataset)],
            'training_time': [training_time],
            'evaluation_time': [evaluation_time],
            'num_params': [num_params],
            'date': [date.strftime("%Y-%m-%d_%H-%M-%S")],
            'accuracy': [eval_metrics['eval_accuracy']],
            'precision': [eval_metrics['eval_precision']],
            'recall': [eval_metrics['eval_recall']],
            'f1': [eval_metrics['eval_f1']],
            'training_loss': [train_result.training_loss],
            'eval_loss': [eval_metrics['eval_loss']],
            'hardware_info': [str(hardware_info)],
            'learning_rate': [learning_rate],
            'batch_size': [batch_size],
            'num_epochs': [epochs],
            'seed': [seed],
        }
        
        data_frame = pd.DataFrame(data)

        if os.path.exists('2. models/runtimes/training/training_info.csv'):
            # If the file exists, load it and append the new data
            existing_df = pd.read_csv('2. models/runtimes/training/training_info.csv')
            data_frame = pd.concat([existing_df, data_frame])

        data_frame.to_csv('2. models/runtimes/training/training_info.csv', index=False)

        # Make predictions
        predictions = trainer.predict(test_dataset)
        logits = predictions.predictions
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        # Save predictions to a CSV file
        filename = f'{self.model_name.replace("/", "-")}_{str(self.adapter)}_{self.column_name}_{date.strftime("%Y-%m-%d")}_predictions.csv'
        predictions_df = pd.DataFrame(probabilities)
        predictions_df.to_csv(f'2. models/predictions/{filename}', index=False)


        if save_model:
            model_dir = '2. models/adapters' if self.adapter else '2. models'
            self.model.save_pretrained(f'{model_dir}/{self.model_name}')

