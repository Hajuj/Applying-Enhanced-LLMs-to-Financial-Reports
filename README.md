# Advanced-Analytics-and-Machine-Learning
University project on Advanced Analytics and Machine Learning (AAML) with the goal of outperforming financial analysts via NLP using their own texts.

# Outperforming Morningstar Analysts: Applying Enhanced LLMs to Financial Reports

This repository is dedicated to the research paper "Outperforming Morningstar Analysts: Applying Enhanced LLMs to Financial Reports," authored by Jonas Gottal and Mohamad Hgog from LMU Munich's Department of Computer Science. This study analyzes financial analyst reports from Morningstar, employing Large Language Models (LLMs) to extract insights that typically surpass the predictive accuracy of human analysts.

## Abstract
This study examines financial analyst reports from Morningstar, revealing two key findings: firstly, analysts appear to select stocks arbitrarily, and secondly, they provide comprehensive textual justifications that enable informed decisions surpassing mere chance. We employ enhanced large language models to efficiently analyze the textual data, facilitating a broad exploration of various pre-trained models to identify the subtle underlying sentiment and extract more value from the reports than the experts themselves.

## Repository Structure
```
/Advanced-Analytics-and-Machine-Learning
|-- data/                # Raw and processed datasets from Morningstar
|-- models/              # Pre-trained models and adapter configurations
|-- evaluation/          # Scripts and notebooks for model evaluation
|-- presentation/        # Slides and other materials for presentations
|-- report/              # LaTeX source files for the research paper
|-- notebooks/           # Jupyter notebooks with data analysis and model training
|-- requirements.txt     # Project dependencies
|-- README.md            # Repository documentation
|-- LICENSE              # License details
|-- build_model.py       # Script to build models
|-- evaluate_model.py    # Script to evaluate models
|-- run.py               # Main script to run analyses
```

## Approach:
### Objective: 
The study aims to evaluate the value of financial analyst reports from Morningstar by using enhanced Large Language Models (LLMs) to analyze the textual data within these reports.

1. Data and Preprocessing: We utilized financial reports from Morningstar, which included detailed information about stocks such as ratings, prices, analyst notes, and other key financial indicators. The data was preprocessed by removing noise and normalizing figures to create a clean dataset for analysis.
1. Model Implementation: The study leveraged the Transformer architecture and its variants, primarily sourced from Hugging Face. We utilized parameter-efficient techniques such as Adapters for fine-tuning pre-trained models specifically to the domain of financial texts, which helps in managing computational resources effectively.
1. Fine-Tuning Strategy: The fine-tuning involved using adapters and Transformer models to analyze sentiment and subtle cues within financial texts. Different configurations of adapters were tested to find the most efficient setup that retained the effectiveness of the underlying model without extensive retraining.
1. Experimental Design: Our methodology included testing various models and configurations independently, assuming parameter independence to facilitate comparability. We explored different text inputs from the reports to determine which sections carried the most predictive power regarding stock performance.


## Results:
1. Model Performance: The results indicated that traditional financial analyst reports and models like XGBoost could not outperform randomness, suggesting they might not contain actionable insights on their own. However, when applying advanced LLMs to the textual data, our models could extract meaningful information that surpassed traditional analysis methods.
![Zero Experiment ROC Curve](https://github.com/trashpanda-ai/Advanced-Analytics-and-Machine-Learning/blob/ddd31424ede190bffc22f1d5721c70d4357f0181/3.%20evaluation/roc_curves/Zero%20Experiment.png?raw=true)

1. Best Inputs and Configurations: The best results were obtained using text from the "FinancialStrengthText" section of the reports, indicating that summaries of financial strength were highly predictive. Adapters, particularly those in certain configurations, proved to be effective in enhancing model performance with less computational overhead.
![First Experiment ROC Curve](https://github.com/trashpanda-ai/Advanced-Analytics-and-Machine-Learning/blob/2498a56414612f90eb08d9548abb8667f38b35a9/3.%20evaluation/roc_curves/First%20Experiment.png?raw=true)
![Second Experiment ROC Curve](https://github.com/trashpanda-ai/Advanced-Analytics-and-Machine-Learning/blob/2498a56414612f90eb08d9548abb8667f38b35a9/3.%20evaluation/roc_curves/Second%20Experiment.png?raw=true)
![Third Experiment ROC Curve](https://github.com/trashpanda-ai/Advanced-Analytics-and-Machine-Learning/blob/2498a56414612f90eb08d9548abb8667f38b35a9/3.%20evaluation/roc_curves/Third%20Experiment.png?raw=true)

1. Comparative Analysis: Finally, we directly fine-tuned the underlying model with our best adapter setup, which only slightly improved the AUC but significantly reduced the model's complexity and resource usage.
![Fourth Experiment ROC Curve](https://github.com/trashpanda-ai/Advanced-Analytics-and-Machine-Learning/blob/71bd69a88764a5958010c133096a41c742edb5e4/3.%20evaluation/roc_curves/Fourth%20Experiment.png?raw=true)
1. Insights via Interpretability: Using tools like Captum, we provided insights into which features were most influential in the model predictions, aiding in the interpretability and trustworthiness of our machine learning solutions.
![Positive Result](https://github.com/trashpanda-ai/Advanced-Analytics-and-Machine-Learning/blob/5da765c6771ffc825dc7bd39d4c6f4b7d4096a4b/5.%20report/pictures/high_1.png?raw=true)

![Negative Result](https://github.com/trashpanda-ai/Advanced-Analytics-and-Machine-Learning/blob/5da765c6771ffc825dc7bd39d4c6f4b7d4096a4b/5.%20report/pictures/low_3.png?raw=true)


Overall, our research demonstrated that employing advanced machine learning techniques, specifically tailored LLMs, on financial texts could significantly outperform traditional methods used by financial analysts, providing a more robust tool for financial decision-making.

## Installation and Usage
To set up and run the analysis, follow these steps:
```bash
git clone https://github.com/trashpanda-ai/Advanced-Analytics-and-Machine-Learning.git
cd Advanced-Analytics-and-Machine-Learning
pip install -r requirements.txt
python run.py
```

For more detailed instructions on running specific parts of the analysis or models, refer to the documentation within each folder. Please note that ```adapters``` (0.1.2) and ```transformers``` sometimes throw warnings during the install process and may need to be repeated.

## License
This project is distributed under the MIT License. See the `LICENSE` file for more information.