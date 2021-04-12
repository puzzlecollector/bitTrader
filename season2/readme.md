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


[Transformer Encoder using close prices and id](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer_id_close.ipynb) 
- The loss was set as MSE 
- Used proportions of positive returns in the predictions as buy quantity 
- Used close prices and the coin id as features. The coin id was scaled by dividing the values by 10. This is because most of the close price values are around 1, whereas the coin id can range up to 9.  

[Transformer Encoder using close prices](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer%20Returns%20Proportions.ipynb). 
- The loss was set as MSE 
- Used proportions of positive returns in the predictions as buy quantity 
- Used close prices only. 
