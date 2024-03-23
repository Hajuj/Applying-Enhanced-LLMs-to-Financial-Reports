# Advanced-Analytics-and-Machine-Learning
University project on Advanced Analytics and Machine Learning (AAML) with the goal of outperforming financial analysts via NLP using their own texts.

# Approach
## Data
Analyst reports from Morningstar: contain both a subjective analyst report (text) on a stock and a quantifiable forecast (numeric)
## Scope 
Use the various text elements to fine-tune different transformer models (maybe modify some parts with adapters from https://docs.adapterhub.ml) and compare to the predictive qualities from the analysts themselves. 

## ToDos
### Prio 1: Basic Model Pipeline (ensemble)
	- Incorporate an easy swap of text fields (and include multiple text fields at once)
	- Incorporate an easy swap of huggingface models (Bert, FinBERT, ...)
	- Incorporate an easy way to include adapters [Adapterhub](https://adapterhub.ml)
	- Save all models (if fine-tuned) and basic metrics (training times, etc)
### Prio 2: Automated Evaluation (parallel)
	- Build custom ROC curves on top of confusion matrix/other evaluation methods and save all metrics
	- Use industry standard data (Stanford sentiment) to benchmark transformer models and training time
	- Compare to Analysts predictions and to actual performance
### Prio 3: Features (parallel)
	- Calculate Returns, Sharpe Ratio and max draw down
	- Normalize the probabilities/Include weights based on normalized probabilities for portfolio
	- Explore with feature attribution methods for the best performing model to gain deeper insights: what part of the input text is the reason for the prediction? (Focus on interpretability and explainable AI) 
### Report
	- Visualize Training/inference times
	- ROC/Evaluation
	- Visualize important input features
	- Explain transformers
	- Explain data set