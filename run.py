import pandas as pd
from build_model import CustomTransformerModel
from evaluate_model import ModelEvaluator

# Load the training and evaluation data

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

# Load the true labels
eval_df = pd.read_csv('1. data/final/eval.csv')
true_labels = eval_df['label']

# Create an instance of ModelEvaluator
evaluator = ModelEvaluator(true_labels, combinations)

# Evaluate the models
evaluator.evaluate()


