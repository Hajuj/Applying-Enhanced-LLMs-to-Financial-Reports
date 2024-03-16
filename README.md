# Advanced-Analytics-and-Machine-Learning
University project on Advanced Analytics and Machine Learning (AAML) with the goal of outperforming financial analysts via NLP using their own texts.

# Approach
## Data
Analyst reports from Morningstar: contain both a subjective analyst report (text) on a stock and a quantifiable forecast (numeric)

## ToDos
- Use the various text elements to fine-tune different transformer models (maybe modify some parts with adapters from https://docs.adapterhub.ml) and compare to the predictive qualities from the analysts themselves. 
	- Incorporate an easy swap of text fields (and include multiple text fields at once)
	- Incorporate an easy swap of huggingface models (Bert, FinBERT, ...)
	- Incorporate an easy way to include adapters
	- Normalise the probabilities
	- Build custom ROC curves
	- Calculate Returns, Sharpe Ratio and max draw down
	- Include weights based on normalised probabilities for portfolio
	- Use industry data (Stanford sentiment) to benchmark transformer models and training time
	- Compare to Analysts predictions and to actual performance
	- Explore with feature attribution methods to gain deeper insights: what part of the input text is the reason for the prediction? (Focus on interpretability and explainable AI) - Compare models with synthetic/other data 
