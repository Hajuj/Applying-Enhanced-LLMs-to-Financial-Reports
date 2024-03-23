# Data Repository
Our data repository is forked from another [data mining project](https://github.com/trashpanda-ai/Analyst-Reports-NLP) and combines ```Analyst reports``` and ```market data```.
## Analyst Reports (Morningstar)
The Analyst Reports are parsed from the british Morningstar website and contain the following variables:

- ```ParseDate```: The date when the parser retrieved the information. (DD/MM/YYYY)
- ```Title```: The title of the analyst report. (String)
- ```CompanyName```: The name of the company. (String)
- ```TickerSymbol```: The ticker symbol (without exchange information) of the underlying stock. (String)
- ```Rating```: The analyst rating of the stock. (Integer)
- ```ReportDate```: The date of the release of the report. (DD/MM/YYYY)
- ```AuthorName```: The name of the author of the report. (String)
- ```Price```: The price of the stock declared in the report. (Float)
- ```Currency```: The given currency of the price. (String)
- ```PriceDate```: The date the price was retrieved from market data (Morningstar). (DD/MM/YYYY)
- ```FairPrice```: The estimated fair price from the analyst. (Float)
- ```Uncertainty```: The company's uncertainty quantified in 'Low', 'Medium', 'High' and 'Very High'.
- ```EconomicMoat```: The economic moat (a business' ability to maintain competitive advantages) quantified in 'Narrow' and 'Wide'.
- ```CostAllocation```:  Capital Allocation refers to the strategic decisions made by a company's management regarding the deployment of funds, focusing on three key aspects: balance sheet management, investment decisions, and shareholder distributions, with the goal of optimizing total shareholder returns -- quantified in 'Poor', 'Standard' and 'Exemplary'.
- ```FinancialStrength```: The Financial Strength rates the ability of a company to make timely and full payment of its obligations to policyholder claims and benefits, as well as financial contract guarantees and benefit obligations -- quantified in 'A', 'B', ... 'F'.
- ```AnalystNoteDate```: The date of the analyst note. (DD/MM/YYYY)
- ```AnalystNoteList```: The analyst note as a list of paragraphs. (String)
- ```BullsList```: A list of arguments for the company. (List of Strings)
- ```BearsList```: A list of arguments against the company. (List of Strings)
- ```ResearchThesisDate```: The date of the research thesis. (DD/MM/YYYY)
- ```ResearchThesisList```: An objective research thesis as a list of paragraphs. (List of Strings)
- ```MoatAnalysis```: Added research on ```EconomicMoat```. (String)
- ```RiskAnalysis```: Research on the company's risk profile. (String)
- ```CapitalAllocation```: Text on the ```CostAllocation``` of the company. (String)
- ```Profile```: Short text on the company profile (summary). (String)
- ```FinancialStrengthText```: Short text on the financial strength of the company (conclusion). (String)


The data is in JSON and has the following format:

```{
    "0P00017BZR": {
        "Index": 0,
        "ParseDate": "24/11/2023",
        "ID": "0P00017BZR",
        "Title": "Virgin Money Meeting Short-Term Targets but Medium-Term ROE Targets ...",
        "CompanyName": "Virgin Money UK PLC",
        "TickerSymbol": "VMUK",
        "Rating": 4,
        "ReportDate": "24/11/2023",
        "AuthorName": "Nathan Zaia",
        "Price": {
            "Value": 146.75,
            "Currency": "GBX",
            "Date": "23/11/2023"
        },
        "FairPrice": 210.0,
        "Uncertainty": "High",
        "EconomicMoat": "None",
        "CostAllocation": "Standard",
        "FinancialStrength": "",
        "AnalystNote": {
            "Date": "24/11/2023",
            "Text": [
                "Virgin Moneys fiscal 2023 preprovision profit missed ...",
                "The earnings miss ...",
                "Shares trade at a ..."
            ]
        },
        "Bulls": [
            "The combination of ... ",
            "With a full-service product range, ...",
            "Legacy customer conduct..."
        ],
        "Bears": [
            "Customers placing less emphasis on ...",
            "Cost savings ...",
            "A desire to grow ... "
        ],
        "ResearchThesis": {
            "Date": "24/11/2023",
            "Text": [
                "Virgin Money UK ...",
                "Acquiring Virgin Money ...",
                "We believe the bank's ...",
                "Opportunities to ... ",
                "The other key part ... "
            ]
        },
        "MoatAnalysis": "Virgin Money UK, or Virgin, ... ",
        "CapitalAllocation": "Our capital allocation rating ...",
        "Overview": {
            "Profile": "Virgin Money UK was ... ",
            "FinancialStrength": "The capital structure ..."
        }
    },
    ...
}
```


## Market Data
The dedicated market data is pulled from yahoo finance via the (unofficial) Python API. We pulled daily (EOD) data on OHLCV (Open High Low Close Volume) 30 (or 60) days before and after the release of the reports. We also included the company's ```Sector```, the	```MarketCap```	and its traded ```Exchange```. Unfortunately the ```TickerSymbol``` of the analyst report is not specific enough. Thats why the [Jupyter notebook '2. Data Merge'](https://github.com/trashpanda-ai/Analyst-Reports-NLP-/blob/e2b421149b506df3004cbe21ee8ec53f33352a56/2.%20Data%20Merge.ipynb) shows a detailed approach how to find the actual ticker symbols.

## Load Data

```
import json
import pandas as pd

# Load the data from the JSON file
with open('Data/Market Data/30d_data/JSON/all_data.json') as f:
    all_data_60 = json.load(f)

# Convert each item in the dictionary back into a DataFrame
all_data_60_df = {key: pd.read_json(value, orient='columns') for key, value in all_data_60.items()}

df = pd.read_csv('Data/final_data.csv', index_col=0)
```


Info: ```final_data.csv``` is a combination of the Morningstar reports and a usable ticker symbol to pull the yahoo data.