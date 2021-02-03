# ESG Reddit Scraper

[![esg_image](https://www.jpmorgan.com/content/dam/jpm/commercial-banking/insights/leadership/esg%20-article-graphic.jpg)](https://www.jpmorgan.com/content/dam/jpm/commercial-banking/insights/leadership/esg%20-article-graphic.jpg)

**Please check [ESG SCRAPER REPORT](https://amirhallak.github.io/ESG_Report/scripts/Report_Notebook.html) for the full report and visualizations.**

The main scripts can be found under [scripts](https://github.com/amirhallak/ESG_Report/blob/main/scripts/).


## Table of Contents
* [Requirments](#Requirments)
* [Imports](#Imports)
* [Configuration](#Configuration)
* [Installation/Usage](#Installation/Usage)

# Requirments
Python 3.7.5 64-bit or equivalant.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [praw](https://praw.readthedocs.io/en/latest/), the Reddit scraper used in this project.

Other modules also required such as [NLTK](https://www.nltk.org/), [seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/), [pandas](https://pypi.org/project/pandas/), and [NumPy](https://pypi.org/project/numpy/).
```bash
pip install praw
pip install nltk
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
```

# Imports
For the [scraper](https://github.com/amirhallak/ESG_Report/blob/main/scripts/reddit_scraper.py) script:
```python
import praw, pandas as pd, datetime
```

For the [merger](https://github.com/amirhallak/ESG_Report/blob/main/scripts/merge_clean.py) script:
```python
import praw, pandas as pd, datetime
```

For the [report notebook](https://github.com/amirhallak/ESG_Report/blob/main/scripts/Report_Notebook.ipynb) script:
```python
import pandas as pd, numpy as np, seaborn as sns, datetime, nltk, re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import matplotlib as mpl
```

# Configuration
This project was more of a report showing, however if intersted in scraping other keywords and subreddits, you need to edit the keywords and subreddits in the [scraper](https://github.com/amirhallak/ESG_Report/blob/main/scripts/reddit_scraper.py) code. They are the first thing after the path declarations. 

Current keywords and subreddits scraped:
```python
# keyword list
keyword_list = [
    'esg', 'ethical investing', 'greenwashing', 'cenovus', 'enbridge',
    'impact investing', 'husky energy','suncor', 'sustainable investing',
    ]

# subreddit list
list_of_subreddits = [
    'FinancialCareers', 'economics', 'Investing',
    'RobinHood', 'WallStreetBets', 'SecurityAnalysis',
    'InvestmentClub', 'StockMarket', 'Stock_Picks',
    'Forex', 'Options', 'FinancialIndependence', 
    'sustainability', 'personalfinance',
    'worldnews', 'finance', 'kickstarter',
    'academia', 'askeconomics', 'politics',
    'askacademia', 'badeconomics','canada', 
    'calgary', 'college', 'news', 'gradschool',
    'technology', 'tech', 'startups',
    'techolitics','futurology', 'renewableenergy','engineering',
    'cryptocurrency', 'environment', 
    'Europe', 'zerowaste', 'green', 'unitedkingdom', 'energy',
    'humanrights', 'climatechange', 'climate'
    ]
```
If the keywords were changed, then the name of the output dataframes under [dateframes](https://github.com/amirhallak/ESG_Report/tree/main/dataframes) will change. So make sure the paths are correct when you want to run the [merger](https://github.com/amirhallak/ESG_Report/blob/main/scripts/merge_clean.py) file. 

# Installation/Usage

### Step 1
* :running: Clone this repo to your local machine using: 

```bash
git clone https://github.com/amirhallak/ESG_Report
```

### Step 2
* Make sure to follow the instructions in the [Configuration](#Configuration) section. 

### Step 3
* Run the scripts in your terminal in order:

```bash
python reddit_scraper.py
python merge_clean.py
```

### Step 4
* You have newly scraped data in the [dataframes](https://github.com/amirhallak/ESG_Report/tree/main/dataframes) folder! 

