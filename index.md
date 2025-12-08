---
layout: default
---

# What is a time series?
A time series is a sequence of data points that are recorded over **consistent intervals** of time. Examples include hourly or daily temperature data, monthly credit card charges, and the annual GDP growth rates. <br>
<br>
Time series modeling is frequently used in **finance** (e.g., stock market forecasting), **economics** (e.g., inflation or CPI projections), **business** (e.g., monthly sales or inventory analysis), **climatology** (e.g. El Niño cycles and global warming trend detection), **health** (e.g. patient vitals or disease outbreak monitoring), and many fields in **science**. <br>
<br>

## Key characteristics
- **Trend** <br>
- **Seasonality** <br>
- **Cyclic** <br>
- **Noise / randomn variation** <br>
- **Stationarity** <br>
<br>

 ![ts_decomposition](assets/css/ts_decomposition.png)<br>
 <sub> Source: [Seasonal Decomposition of Your Time-Series](http://alkaline-ml.com/pmdarima/1.8.1/auto_examples/arima/example_seasonal_decomposition.html) </sub>
 <br>

## Common issues with time series modeling
- **Missing values and random fluctuations** require data imputation or noise filtering. <br>
- **Non-stationarity and heteroskedasticity** require transformations like differencing or decomposition for the former, and log transformation for the latter.  <br>
- **Structural breaks** or regime changes when there are sudden shifts in the underlying pattern. <br>
<br>

## Examples of helpful uses for our everyday life
- **Forecasting** of future events and trends allows for better planning and more efficient risk management or mitigation strategies. For instance, use of weather and traffic forecasts to plan trips.<br>
- **Pattern recognition** helps earlier diagnosis and focused treatment. <br>
- **Anomaly detection** uncovers fradulent credit card charges or suspicious bank activities. <br>
<br>
<br>

## What makes time series different from "regular" data?

|                      | Time Series       | "Regular" Data       |
|:---------------------|:------------------|:----------------------
| Order & Time         | Essential         | Unimportant usually  |   
| Data Point Dependency| Dependent         | Independence assumed |
| Purpose              | Forecasting       | Understanding        |
| Analysis             | Specialized models| Standard regression  | 

<br>
<br>

## Ways to model time series in machine learning
- **Traditional methods** include 
- **Newer methods**

# Case study: predicting Covid-19 cases in Taiwan
## Background
- Over [7 million deaths](https://data.who.int/dashboards/covid19/deaths?m49=001&n=c) and nearly [779 million cases](https://data.who.int/dashboards/covid19/cases?m49=001&n=c) from COVID-19 have been recorded globally as of November 2025.<br>
<br>
- Taiwan’s early response resulted in low numbers of cases and deaths until early 2022.<br>
<br>
- From March 2022, Taiwan gradually lifted pandemic-related restrictions after reaching 79% in vaccination coverage.<br>
<br>
- Shortly after, there were sharp increases in deaths and cases. Three epidemic waves were observed from 4/17/2022 to 3/18/2023, each seemed “flatter” than the previous.<br>
<br>
![Covid_Waves_Taiwan](/assets/css/Covid_Waves_Taiwan.png)<br>
<br>
<br>

## A tale of two countries
Every country has a different epidemic curve for Covid-19 due to different values, policies, preventive measures, availability of vaccines, etc. The juxtaposition below is a visual representation of said differences. <br>
<br>
![US_vs_Taiwan](assets/css/US_vs_Taiwan.png)<br>
 <br>

## The research question
Given the uniqueness of Taiwan's epidemic waves of Covid-19, how well can machine learning methods forecast Covid cases in Taiwan? <br>
<br>

# Training a LSTM model from scratch

## What is a LSTM model?

## Advantages of LSTM models



## LSTM model setup
Predict Covid cases using Covid cases from the past 5 days
The input-output matrix would look like this: <br>

| Input   | Output| 
|:--------|:------|
|1 2 3 4 5| 6     |
|2 3 4 5 6| 7     | 
|3 4 5 6 7| 8     | 

<br>
Or, if it is easier to conceptualize in terms of day of the week:<br>
<br>

| Input              | Output| 
|:-------------------|:------|
|MON TUE WED THU FRI | SAT   |
|TUE WED THU FRI SAT | SUN   | 
|WED THU FRI SAT SUN | MON   | 

<br>

## LSTM modeling results comparison
Due to the magnitude of Covid cases, the daily totals were "normalized" to range between 0 and 10 by dividing by 10,000 before training. The predicted results were then multiplied by 10,000 to get back to the original unit (number of cases).<be>

To test the LSTM model, two different datasets were used for training: daily temperatures (in Celsius) of Park Slope, NY from 2010-01-01 to 2025-09-30, and daily Covid cases in Taiwan from 2022-04-17 to 2023-03-18. <br>
<br> 
Below is a comparison table of the two datasets on their respective sample size, range of values, training specifications, and minimum difference between predicted and actual values as a percentage of the actual value.<br>
<br>

|                      | Temperature    | Covid Cases   |
|:---------------------|:---------------|:--------------|
| No. of Samples       | 5,752          | 336           |
| Predicted Value Range| 19.24 ~ 22.42  | 12,330        |
| Learning Rate        | 0.01           | 1.0           |
| Epochs               | 200            | 200           |
| Min Error Magnitude  | 0.4%           | 21%           |
| Max Error Magnitude  | 21%            | 54%           |

<br>
<br>

### Good for predicting temperatures but not for Covid cases?
Compared to its performance on the temperature dataset, the LSTM model completed missed the mark in predicting Covid cases in Taiwan during the study period. Why is this the case (pun intended)?<br>
<br>
![Temp_vs_Covid_Cases](assets/css/LSTM_Comparison.png) <br>
 <br>

## Reasons for poor performance on Covid cases prediction
- **Dataset is too small** (only 336 observations). The five-layer LSTM model used in this study has over 17,425 parameters, and a general rule of thumb for neural networks is to have at least 10 to 20 samples per parameter.<br>
<br>  
- **Not enough repeated patterns**. Without repeated patterns, the "long-term memory" component of LSTM offers little value.<br>
<br>
- **Big difference in the magnitude** of daily cases betweening the training dataset and the testing dataset.<br>
<br>

![Temp_vs_Covid_Cases](assets/css/Temp_vs_Covid_Cases.png)<br>
 <br>

# Fine-tuning a pretrained model
Conceptually, one can copy and paste the small dataset, say, 100 times to create artificial repeated patterns and see if the model does any better. However, this can create other modeling and analysis issues. <br>
<br>
So, what if there is already a model that was pretrained on millions of samples? Would the pretrained model perform better than the LSTM model if the study's small dataset is provided as the context for fine-tuning? <br>
<br>

## Amazon Chronos fine-tuning specification

<br>

<br>

```python
!pip install git+https://github.com/amazon-science/chronos-forecasting.git
```


<br>

## Results 
Fine-tuning Chronos pretrained model 
<br>

| Covid Cases          | Chronos Fine-Tuning | LSTM Training  |
|:---------------------|:--------------------|:---------------|
| No. of Samples       | 326                 | 336            |
| Predicted Value Range| 7,957 ~ 9,776       | 12,330         |
| Learning Rate        | na                  | 1.0            |
| Epochs               | na                  | 200            |
| Min Error Magnitude  | 1%                  | 21%            |
| Max Error Magnitude  | 37%                 | 54%            |

<br>

![Chronos_Results](assets/css/Chronos_Results.png)<br>
 <br>

![Chronos_Plot](assets/css/Chronos_Plot.png)<br>

 <br>



# Key takeaways

# Future research
1. Multivariable model in which Covid cases, temperature, vaccination rate would be used to predict Covid deaths. <br>
<br>
2. Adding sine and cosine functions can be added to further remove seasonality for datasets with many repeated pattern, such as the temperature data. <br>
<br>
<br>

# Resources
- [Temp_11215.csv](assets/css/Temp_11215.csv) <br>
- [Daily_Covid_Cases_Taiwan.csv](assets/css/Daily_Covid_Cases_Taiwan.csv) <br>
- [LSTM Time Series Forecasting Tutorial in Python](https://www.youtube.com/watch?v=c0k-YLQGKjY)<br>
- [Install Chronos AI Models for Time Series Forecasting](https://www.youtube.com/watch?v=WxazoCVkBhg)<br>
- [Amazon Chronos-T5 (Tiny)](https://huggingface.co/amazon/chronos-t5-tiny) <br>
<br>