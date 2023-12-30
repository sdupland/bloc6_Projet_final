# bloc6 Projet final

Our graal : Predict the price of bitcoin for the next day !

We usually take advantage of models capable of detecting patterns on time series (lstm, arima, prophet…).
Often models are univariate, based on the historical price evolution of the asset, and sometimes multivariate with a few features.

We propose here to treat the subject in a different way, considering that historical prices can’t help us to find the futur price.
As we said in Finance, and most of all on stock market : “ Past performance does not guarantee future performance “.

To do that, we transformed this time series problem in a supervised machine learning task through different steps

1) Get raw data and create our dataset by feature engineering

Raw data were got via a yahoo finance api (yfinance) over a period of 5 years (daily basis, 5 features collected :open, high, low, close, volume).

We add around :
- 70 features based on technical indicators (RsI, Bollinger Bands) calculted from raw data,
- calendar features (month, day etc.)
- and targets value (close price of the next day for example if we want to make prediction at this horizon of time

We made a first feature selection in order to eliminate those who had the lower correlation coefficient.

2) Preprocess the data

We made a data normalization with standardscaler rather than Minmaxscaler.
Three datasets were created :
- training 70%, validation 15% and test 15%,
- with a break-period between each one of 20 days

At least features and target were separated.

3) Choose and train models

We choose three models :
- Linear regression,
- Regularization with Elasticnet
- XGBoost

To measure performance, we retained the MAPE (mean absolute percent error) for two reasons :
- Error in amount is difficult to interpret as bitcoin price goes for a large range of value over the period (from 4 000 € to 40 000 € today)
- in a business approach (for a trading strategy for example), this indicator is more relevant to our point of view

We worked on optimization of performance by using feature selection (with RFE) and fine tuning.

Models were evaluated on test set.

4) Deploy our work : create a dashboard

We used streamlit and heroku for technical aspects

The goal of this dashboard is  multiple.

First to give some insights on the bitcoin based on historical data. To do that, you will find on some previews on data used in this dashboard, split into different parts :
- raw data from yfinance (bitcoin prices and volume)
- a set of technical indicators calculated from the raw data, and grouped in a few categories according to the main existing families (trend, volume, oscillator ..)

Secondly to facilitate the comprehension of Bitcoin evolution ovee the period through some graphs about bitcoin prices and technical indicators.

Thirdly to present the work done on data modelisation through the use of machine learning models in order to predict the price of Bitcoin.
You will find in p3 a short presentation of different models, and the results we got.

Fourth to predict the price of the bitcoin regarding different time horizons, and models.

At least, we made a synthesis of this work which contains :
- some limitations and reflections about the job done
- various ways to improve it
