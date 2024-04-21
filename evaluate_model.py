from datetime import datetime
import os
import glob
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


"""
import pandas as pd
import glob
# List of [model_name, adapter, column_name] combinations
combinations = [
    ['bert-base-uncased', True, 'Analyst Note', 'adapter_config'],
    # Add more combinations as needed
]

# Load the true labels
eval_df = pd.read_csv('1. data/final/eval.csv')
true_labels = eval_df['label']

# Create an instance of ModelEvaluator
evaluator = ModelEvaluator(true_labels, combinations)

# Evaluate the models
evaluator.evaluate()
"""

class ModelEvaluator:
    def __init__(self, true_labels, combinations):
        self.true_labels = true_labels
        self.combinations = combinations
        self.results = []


    def evaluate(self):
        for model_name, adapter, column_name, adapter_config in self.combinations:
            # Construct the filename and use regex to allow all dates in the format {date.strftime("%Y-%m-%d")}
            # Construct the filename pattern
            filename_pattern = f'{model_name.replace("/", "-")}_{str(adapter)}_{column_name}_*_predictions.csv'

            # Find all matching files
            matching_files = glob.glob(f'2. models/predictions/{filename_pattern}')

            date = datetime.now()

            # Load the predictions for each matching file
            for file in matching_files:
                predictions_df = pd.read_csv(file)
                # Calculate the AUC score
                auc_score = roc_auc_score(self.true_labels, predictions_df.iloc[:, 1])

                # Save the results
                self.results.append([model_name, adapter, str(adapter_config), column_name, auc_score, date.strftime("%Y-%m-%d")])

                # Generate the ROC curve
                fpr, tpr, _ = roc_curve(self.true_labels, predictions_df.iloc[:, 1])
                plt.figure()
                plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_score)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                #filename = f'{model_name.replace("/", "-")}_{str(adapter)}_{column_name}'
                plt.savefig(f'3. evaluation/roc_curves/{model_name.replace("/", "-")}_{str(adapter)}_{column_name}_{date.strftime("%Y-%m-%d")}.png')

        # Save the results to a CSV file    
        # Create a DataFrame with the results
        results_df = pd.DataFrame(self.results, columns=['model_name', 'adapter', 'adapter_config', 'column_name', 'auc_score', 'date'])

        # Check if the CSV file already exists
        if os.path.exists('3. evaluation/auc_scores/auc_scores.csv'):
            # Load the existing data
            existing_df = pd.read_csv('3. evaluation/auc_scores/auc_scores.csv')

            # Append the new results
            results_df = pd.concat([existing_df, results_df])

        # Sort the DataFrame by 'auc_score' in descending order
        results_df = results_df.sort_values('auc_score', ascending=False)

        # Save the results to a CSV file
        results_df.to_csv('3. evaluation/auc_scores/auc_scores.csv', index=False)

    def build_roc_curve(self, n):
        top_models = pd.read_csv('3. evaluation/auc_scores/auc_scores.csv').head(n)
        plt.figure(figsize=(10, 8))

        for index, row in top_models.iterrows():
            model_name = row['model_name']
            adapter = row['adapter']
            column_name = row['column_name']
            date = row['date']
            filename = f'{model_name.replace("/", "-")}_{adapter}_{column_name}_{date}_predictions.csv'
            predictions_df = pd.read_csv(f'2. models/predictions/{filename}')
            auc_score = roc_auc_score(self.true_labels, predictions_df.iloc[:, 1])
            fpr, tpr, _ = roc_curve(self.true_labels, predictions_df.iloc[:, 1])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'3. evaluation/roc_curves/top_{n}_models.png')