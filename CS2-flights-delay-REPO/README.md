# CS2-flights-delay

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Statement

When planning flights, travelers often create itineraries with both direct flights and multiple connections across various airports, sometimes involving different airlines. Airlines and airports may experience varying rates of delays and cancellations, influenced by factors such as time of day, season, weather conditions, and traffic levels. Travelers may have strict or flexible arrival requirements and may wish to estimate the likelihood of timely arrival or anticipate potential delays with a certain level of confidence.

This project addresses the question: How can travelers predict the likelihood and duration of flight delays with a specified level of accuracy to meet specific arrival time requirements?

The US Domestic Flights Delay (2013-2018) dataset, sourced from the U.S. Office of Airline Information, Bureau of Transportation Statistics (BTS), provides comprehensive data on scheduled and actual departure and arrival times. Covering flights from 2014 to 2018, it includes details such as date, time, origin, destination, airline, distance, and delay status. (Source: [Kaggle](https://www.kaggle.com/datasets/gabrielluizone/us-domestic-flights-delay-prediction-2013-2018))

This project aims to develop a predictive model that estimates the likelihood and duration of flight delays to meet specific arrival requirements. Leveraging historical flight data and machine learning, the model will help travelers better plan their trips. By predicting potential delays, travelers can make informed decisions to reduce the risk of missed connections and optimize travel plans.

## Data Collection and Cleaning

The dataset includes 12 files per year (60 total), each with ~500,000 records and 110 columns, initially estimated at 81 GB. Following analysis, 20 original features were selected and 12 new ones constructed by modifying existing data. To optimize memory, columns were assigned appropriate data types, eliminating the ‘object’ type entirely.

Actual Arrival Delay was chosen as the dependent variable for the prediction model. Significant work was done to verify date and time accuracy:
*	Converted times from ‘hhmm’ to minutes from the start of the day.
*	Calculated dates for scheduled arrival, actual departure, and actual arrival using flight durations and delays.
*	Adjusted all times to UTC based on airport IATA codes to ensure consistency.
*   UTC datetimes were used to verify the consistency of Actual Arrival Delays in the dataset.

Key findings:
*	Pandas datetime columns can only store data with the same time zone. Mixing time zones causes automatic conversion to object, increasing memory usage.
*	Avoiding object data types is crucial. All 'object' and some 'integer' data (e.g., flight numbers) was converted to categorical types for efficiency.
*	When assigning a pd.datetime64 slice to a pd.Series with NaT values, ensure the destination Series has the same time zone as the slice beforehand to avoid unwanted conversion to 'object' type.
*	Concatenating datasets with categorical data can unintentionally convert them to object, requiring post-concatenation checks.

The dataset underwent thorough verification for consistency in Actual Arrival Delays. Records with unresolved inconsistencies, such as large mismatches between elapsed and scheduled flight times, were removed (less than 2% of the dataset).

The cleaned dataset, containing about 30 million records, was exported in .pickle format, reducing memory usage from an initial 81 GB to under 3 GB.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA)

During the EDA stage, we visually observed variations in the mean Actual Arrival Delays across months and weekdays; however, hypothesis testing revealed no statistically significant difference between these time periods. However, Chi-square tests confirmed a significant variation in delays across different airports and airlines at the 5% significance level. Additionally, both visualizations and Chi-square tests highlighted a strong relationship between Actual Arrival Delay and departure/arrival time blocks.

Analysis of flight cancellations revealed no significant correlation with timing, airport, or airline. Other factors, such as Actual Elapsed Time and Arrival Delay for diverted flights, were found to have no predictive value for the model, as they cannot be anticipated during itinerary planning.

The Kolmogorov-Smirnov test confirmed that the target variable, Actual Arrival Delay, is right-skewed and does not follow a normal distribution. Logarithmic and Box-Cox transformations did not improve the distribution. All independent variables with a relationship to the target are categorical and will be treated accordingly in the model design phase.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         flights-delay and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── flights-delay   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes flights-delay a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

