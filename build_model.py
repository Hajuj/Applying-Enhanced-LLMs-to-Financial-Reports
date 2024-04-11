import torch
import os
import pandas as pd
import numpy as np
import time
import platform
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
from transformers import set_seed, TrainingArguments, Trainer, EvalPrediction, AutoTokenizer
# from transformers import set_seed, TrainingArguments, AdapterTrainer, Trainer, EvalPrediction, AutoModelWithHeads, AdapterConfig, AutoTokenizer



""" # Load the training and evaluation data

# List of [model_name, adapter_name, column_name] combinations
combinations = [
    ['bert-base-uncased', 'sst-2', 'Analyst Note'],
    # Add more combinations as needed
]

# Load the training and evaluation data
train_df = pd.read_csv('1. data/final/train.csv')
eval_df = pd.read_csv('1. data/final/eval.csv')

for model_name, adapter_name, column_name in combinations:
    # Create an instance of CustomTransformerModel
    model = CustomTransformerModel(model_name=model_name, adapter_name=adapter_name, column_name=column_name)

    # Build the model
    model.build_model()

    # Train the model
    model.train(train_df, eval_df, epochs=3, batch_size=32, learning_rate=2e-5, seed=42, save_model=False)

"""

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        label = torch.tensor(self.labels[idx])
        return inputs, label

class CustomTransformerModel:
    def __init__(self, model_name: str, adapter_name: str = None, column_name: str = 'text'):
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.column_name = column_name
        self.model = None
        self.tokenizer = None

    def build_model(self):
        self.model = AutoModelWithHeads.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.adapter_name:
            # Add a new adapter
            self.model.add_adapter(self.adapter_name, AdapterConfig())

    def compute_metrics(self, eval_pred: EvalPrediction):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted'),
        }


    def train(self, df: pd.DataFrame, eval_df: pd.DataFrame, epochs: int = 1, batch_size: int = 32, learning_rate: float = 1e-4, seed: int = 42, save_model: bool = False):
        set_seed(seed)
        if self.model is None or self.tokenizer is None:
            raise Exception("Model is not built. Call build_model first.")

        # Assume that the DataFrame has a 'label' column for labels
        train_dataset = TextDataset(df[self.column_name], df['label'], self.tokenizer)
        eval_dataset = TextDataset(eval_df[self.column_name], eval_df['label'], self.tokenizer)


        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=epochs,              # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            learning_rate=learning_rate,               # learning rate
            weight_decay=0.01,               # strength of weight decay
            logging_dir='2. models/logs',            # directory for storing logs
        )

        
        if self.adapter_name:
            trainer = AdapterTrainer(
                model=self.model,   # the instantiated 🤗 Transformers model to be trained
                args=training_args, # training arguments, defined above
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self.compute_metrics,
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self.compute_metrics,
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

        # Create a DataFrame and save it as a CSV file
        data = {
            'model_name': [self.model_name],
            'adapter_name': [self.adapter_name],
            'column_name': [self.column_name],
            'training_data_length': [len(train_dataset)],
            'test_data_length': [len(eval_dataset)],
            'training_time': [training_time],
            'evaluation_time': [evaluation_time],
            'num_params': [num_params],
            'date': [datetime.now()],
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
        }
        
        data_frame = pd.DataFrame(data)

        if os.path.exists('2. models/runtimes/training/training_info.csv'):
            # If the file exists, load it and append the new data
            existing_df = pd.read_csv('2. models/runtimes/training/training_info.csv')
            data_frame = pd.concat([existing_df, data_frame])

        data_frame.to_csv('2. models/runtimes/training/training_info.csv', index=False)

        # Make predictions
        predictions = trainer.predict(eval_dataset)

        # Save predictions to a CSV file
        filename = f'{self.model_name}_{self.adapter_name or "no_adapter"}_{self.column_name}_predictions.csv'
        predictions_df = pd.DataFrame(predictions.predictions, columns=['prediction'])
        predictions_df.to_csv(f'2. models/predictions/{filename}', index=False)

        if save_model:
            model_dir = '2. models/adapters' if self.adapter_name else '2. models'
            self.model.save_pretrained(f'{model_dir}/{self.model_name}')

