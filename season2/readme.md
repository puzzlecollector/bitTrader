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

[Transformer Time2Vec using Open, high, low, close to predict open](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer_OHLC.ipynb)
- Same as above, but uses more features. 
- Scores $20637.61538 on the public leaderboard. 
- Also during the experimentation, I tried using instance norm but it degraded performance, so I switched to batchnorm.  
- Weird thing: the MAPE loss was higher than the previous model that scored $19,407.38464, but somehow the leaderboard score was higher. 
- MAPE loss was 1.01, whereas before it was 0.97. 

[Transformer Time2Vec using both price and volume](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer_OHLC_with_volumes.ipynb) 
- Same model as above, but also incorporates volume features (trades, volume, quote_av etc) 
- passes each through a different transformer network then blends the two results. 
- MAPE of 0.98. 
- Scores $22649.46664 on the public leaderboard.   

[Transformer Time2Vec with GRU trained at the same time](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer_GRU_OHLC_with_volume.ipynb) 
- It is a large model and recoreded the lowest MAPE of 0.95.  
- However, it scores $17779.5271222103 on the public leaderboard. It is much lower than our current best. 
- I suspect that I stopped training a bit too ealry (it seemed to be improving even after 20 epochs). 
- We can experiment more with large models like these. 

[Transformer Time2Vec with noise augmentation](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer_OHLC_volume_data_augmentation.ipynb) 
- With random noise augmentation
- Records a score of $19972.9108932918

[Transformer Time2Vec with additional Moving Average Features](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer%20feature%20engineering%20moving%20average%20features.ipynb) 
- Added open ma5, open ma10, open ma20, open ma60, open ma120 and same for volume as additional features 
- Processed each separately using separate transformers then merged results 
- Does not work very well. Scores pretty low in the $16000 range in the public leaderboard.  

[Using data augmentation especially noise addition and interpolation](https://github.com/puzzlecollector/bitTrader/blob/main/season2/decaying_noise_interpolation_augmentation.ipynb) 
- Used keras generator 
- Training time takes too long and the model seems to have difficulty learning, so I stopped training 
- Did not get to submit.  

[Transformer with sinusoidal positional encoding](https://github.com/puzzlecollector/bitTrader/blob/main/season2/Transformer_positional_encoder.ipynb) 
- Trained a model with N=120 window and sinusoidal positional encoding 
- Max ensembled with Transformer Time2Vec using both price and volume 
- Our current best $26206.23078 on the public leaderboard. 

[Transformer cross attention at the end](https://github.com/puzzlecollector/bitTrader/blob/main/season2/transformer_attn.ipynb) 
- Cross attention 
- Records $21633 alone on the leaderboard 
- seems promising?  


# References 
- Time2Vector 
- Methods of time series prediction 
- Wavenet
