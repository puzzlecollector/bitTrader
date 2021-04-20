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

[Transformer Time2Vec Close only](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer%20Time2Vec%20Close%20Only.ipynb) 
- Transformer 3 layers, with Time2Vec, close prices only
- Prices connected to sequence. 
- Seems pretty promising. Perhaps this is the method we should use.   


[Transformer Time2Vec N+k1 N+k2 method](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer_Time2Vec_Close_60_and_120_point_predictions.ipynb) 
- This method does not work! I thought it would be better than a simple N+K method. 
- I combine predictions from N+60 and N+120. It does not work .......  


[Transformer Time2Vec using Open and Close to predict Open](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer%20Open%20Close%20predict%20Open.ipynb) 
- This method obtains the best result on the public leaderboard as of 2021/04/20.  
- It scores $19,407.38464 on the public leaderboard. It uses the open and the close features to predict the open features using the N+K prediction method. It also uses Time2Vector layer during training.  
- Reduced lookback window from N = 60 to N = 30.  
