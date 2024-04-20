import os
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


"""
import pandas as pd
# List of [model_name, adapter, column_name] combinations
combinations = [
    ['bert-base-uncased', True, 'Analyst Note'],
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
        for model_name, adapter, column_name in self.combinations:
            # Construct the filename
            filename = f'{model_name.replace("/", "-")}_{str(adapter)}_{column_name}_predictions.csv'

            # Load the predictions
            predictions_df = pd.read_csv(f'2. models/predictions/{filename}')

            # Calculate the AUC score
            auc_score = roc_auc_score(self.true_labels, predictions_df['prediction_1'])

            # Save the results
            self.results.append([model_name, adapter, column_name, auc_score])

            # Generate the ROC curve
            fpr, tpr, _ = roc_curve(self.true_labels, predictions_df['prediction_1'])
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_score)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            filename = f'{model_name.replace("/", "-")}_{str(adapter)}_{column_name}'
            plt.savefig(f'3. evaluation/roc_curves/{filename}.png')

        # Save the results to a CSV file
        
        
        # Create a DataFrame with the results
        results_df = pd.DataFrame(self.results, columns=['model_name', 'adapter', 'column_name', 'auc_score'])

        # Check if the CSV file already exists
        if os.path.exists('3. evaluation/auc_scores.csv'):
            # Load the existing data
            existing_df = pd.read_csv('3. evaluation/auc_scores.csv')

            # Append the new results
            results_df = pd.concat([existing_df, results_df])

        # Sort the DataFrame by 'auc_score' in descending order
        results_df = results_df.sort_values('auc_score', ascending=False)

        # Save the results to a CSV file
        results_df.to_csv('3. evaluation/auc_scores.csv', index=False)

##TODO: Add method to build a ROC curve for the top n models in the list
        