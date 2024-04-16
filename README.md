# Advanced-Analytics-and-Machine-Learning
University project on Advanced Analytics and Machine Learning (AAML) with the goal of outperforming financial analysts via NLP using their own texts.

# Approach
## Data
Analyst reports from Morningstar: contain both a subjective analyst report (text) on a stock and a quantifiable forecast (numeric)
## Scope 
Use the various text elements to fine-tune different transformer models (maybe modify some parts with adapters from https://docs.adapterhub.ml) and compare to the predictive qualities from the analysts themselves. 

## ToDos
### Prio 1: Basic Model Pipeline (ensemble)
- ~~Incorporate an easy swap of text fields (and include multiple text fields at once)~~
- ~~Incorporate an easy swap of huggingface models (Bert, FinBERT, ...)~~
- ~~Incorporate an easy way to include adapters [Adapterhub](https://adapterhub.ml)~~
- ~~Save all models (if fine-tuned) and basic metrics (training times, etc)~~
- Data Preprocessing: non-deterministic reproducible train test split (Deadline: 17.04.2024)
- Incorporate current version of Jupyter Notebook in python code and update requirements if necessary + Debug with distilbert and adapter and make sure all exports, files and models are saved in a organized manner (incl predictions as probabilities for ROC curves) (Deadline: 20.04.2024)
- Create a list of potential models/adapters that could be used for financial texts/classification of sentiment (Deadline: 21.04.2024)

### Prio 2: Automated Evaluation (parallel)
- ~~Build custom ROC curves on top of confusion matrix/other evaluation methods and save all metrics~~
- ~~Show why adapters are useful: no (huge) loss in quality while training time is significantly reduced~~
	- ~~Use huggingface sst2 to demonstrate (ROC curves + training time)~~
	- (Only if lots of time: Use various models and compare with the same model incl dedicated adapter)
- Run for analyst note and other text columns and use as label to predict the moving average (see old project) (Deadline: 22.04.2024)
### Prio 3: Features (parallel)
- Single out best model/adapters, discard efficiency and try to optimize
- Finish automated evaluation plots for top $n$ models
- Benchmarking:
	- Add xgBoost based on sector, author name etc
	- Add xgBoost based on analysts fairprice (baseline)
- Explore with feature attribution methods for the best performing model to gain deeper insights: what part of the input text is the reason for the prediction? (Focus on interpretability and explainable AI) (Use e.g. Captum to find important parts of text that lead to the highest probabilities of classifiers)

### Report, presentation & Markdown file
- Visualize Training/inference times
- ROC/Evaluation
- Visualize important input features
- Explain transformers
- Explain data set

### Prio 4: If lots of time
- Build portfolio backtesting: Returns, Sharpe Ratio and max draw down and include weights based on normalized probabilities for portfolio
- Show why adapters are useful: Use various models and compare with the same model incl dedicated adapter