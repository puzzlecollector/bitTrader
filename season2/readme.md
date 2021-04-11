# BitTrader Season 2 

Continuation of season 1. 

We ranked 4th place for season 1. We will try to further develop this methodology for season 2. 


[Transformer Encoder using returns buy-quantity method](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer%20close%20prices%20different%20buy%20quantity%20method.ipynb)
- The loss was set as MAPE instead 
- Used the proportions of positive returns as buy quantity 
- Was not as successful as the transformer using proportions of values higher than buy price (and using mse loss) 

[GRU using 20 minute MA smoothed values](https://github.com/puzzlecollector/bitTrader/blob/main/season2/GRU_20_mins_MA_Prediction.ipynb)
- The loss was set as MAPE instead 
- Used proportions of values higher than buy price
- Used 20 minute Moving Average smoothed values as input 
- Was not as successful as the GRU that used the original time series. 

