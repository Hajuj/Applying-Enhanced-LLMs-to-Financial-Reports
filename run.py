import warnings

import pandas as pd
from adapters import ConfigUnion, PrefixTuningConfig, SeqBnConfig, LoRAConfig, MAMConfig, UniPELTConfig

from build_model import CustomTransformerModel
from evaluate_model import ModelEvaluator

warnings.filterwarnings('ignore')

models_to_test = ["nickmuchi/sec-bert-finetuned-finance-classification",
                  "ProsusAI/finbert",
                  "ahmedrachid/FinancialBERT-Sentiment-Analysis",
                  "yiyanghkust/finbert-tone",
                  "nickmuchi/deberta-v3-base-finetuned-finance-text-classification",
                  "soleimanian/financial-roberta-large-sentiment",
                  "bardsai/finance-sentiment-pl-fast",
                  "RashidNLP/Finance-Sentiment-Classification",
                  "siebert/sentiment-roberta-large-english",
                  "kwang123/bert-sentiment-analysis",
                  "distilbert/distilbert-base-uncased-finetuned-sst-2-english"]

adapter_config_list = [LoRAConfig(), UniPELTConfig(), MAMConfig(), ConfigUnion(LoRAConfig(r=8, use_gating=True),
                                                                               PrefixTuningConfig(prefix_length=10,
                                                                                                  use_gating=True),
                                                                               SeqBnConfig(reduction_factor=16,
                                                                                           use_gating=True)), ]

# List of [model_name, adapter_name, column_name] combinations
# combinations = [
#     ["bardsai/finance-sentiment-pl-fast", True, 'AnalystNoteList', adapter_config_list[0]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'AnalystNoteList', adapter_config_list[0]],
#     ["yiyanghkust/finbert-tone", True, 'AnalystNoteList', adapter_config_list[0]],
#     ["nickmuchi/deberta-v3-base-finetuned-finance-text-classification", True, 'AnalystNoteList',
#      adapter_config_list[0]],
#     ["ProsusAI/finbert", True, 'AnalystNoteList', adapter_config_list[0]],
#     ["ahmedrachid/FinancialBERT-Sentiment-Analysis", True, 'AnalystNoteList', adapter_config_list[0]],
#     ["soleimanian/financial-roberta-large-sentiment", True, 'AnalystNoteList', adapter_config_list[0]],
#     ["RashidNLP/Finance-Sentiment-Classification", True, 'AnalystNoteList', adapter_config_list[0]],
#     ["siebert/sentiment-roberta-large-english", True, 'AnalystNoteList', adapter_config_list[0]],
#     ["kwang123/bert-sentiment-analysis", True, 'AnalystNoteList', adapter_config_list[0]],
#     ["distilbert/distilbert-base-uncased-finetuned-sst-2-english", True, 'AnalystNoteList', adapter_config_list[0]],
# ]

# Fine-tuning adapter for best two models
# combinations = [
#     ["yiyanghkust/finbert-tone", True, 'AnalystNoteList', adapter_config_list[1]],
#     ["yiyanghkust/finbert-tone", True, 'AnalystNoteList', adapter_config_list[2]],
#     ["yiyanghkust/finbert-tone", True, 'AnalystNoteList', adapter_config_list[3]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'AnalystNoteList', adapter_config_list[1]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'AnalystNoteList', adapter_config_list[2]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'AnalystNoteList', adapter_config_list[3]],
# ]

# List for different input columns, for the best model with best adapterconfig:
# BullsList, BearsList, ResearchThesisList, MoatAnalysis, RiskAnalysis, CapitalAllocation, Profile, FinancialStrengthText
# combinations = [
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'BullsList', adapter_config_list[2]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'BearsList', adapter_config_list[2]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'ResearchThesisList', adapter_config_list[2]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'MoatAnalysis', adapter_config_list[2]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'RiskAnalysis', adapter_config_list[2]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'CapitalAllocation', adapter_config_list[2]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'Profile', adapter_config_list[2]],
#     ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'FinancialStrengthText', adapter_config_list[2]],
# ]

# List for fine-tuning LLM and adapter
combinations = [
    ["nickmuchi/sec-bert-finetuned-finance-classification", True, 'FinancialStrengthText', adapter_config_list[2]],
    ["nickmuchi/sec-bert-finetuned-finance-classification", False, 'FinancialStrengthText', adapter_config_list[2]],
]

# Load the training and evaluation data
train_df = pd.read_csv('1. data/final/train.csv')
dev_df = pd.read_csv('1. data/final/dev.csv')
test_df = pd.read_csv('1. data/final/test.csv')

for model_name, adapter, column_name, adapter_config in combinations:
    # Create an instance of CustomTransformerModel
    model = CustomTransformerModel(model_name=model_name, adapter=adapter, column_name=column_name,
                                   adapter_config=adapter_config)

    # Build the model
    model.build_model()

    # Train the model
    model.train(train_df, dev_df, test_df, epochs=5, batch_size=16, learning_rate=1e-4, seed=42, save_model=False)

test_df = pd.read_csv('1. data/final/test.csv')
true_labels = test_df['Label']

# Create an instance of ModelEvaluator
evaluator = ModelEvaluator(true_labels, combinations)

# Evaluate the models
evaluator.evaluate()
evaluator.build_roc_curve(5)
