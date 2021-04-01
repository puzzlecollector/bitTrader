[competition link](https://dacon.io/competitions/official/235709/data/) 

[A more detailed explanation of the competition](https://dacon.io/competitions/official/235709/talkboard/402666?page=1&dtype=recent) 


## Problem Setup 
We are given a certain bitcoin, with chart information for the first 23 hours (or 1380 minutes. Time series data is recorded every minute). 
It is stated that we bought a certain quantity of that coin at the 23rd hour. We need to sell the coin in the next 2 hours.  
The problem is about figuring out two things 
1. How much should be buy at the 23rd hour? 
2. When should we sell the coin? (it has to be sold in the next 2 hours).  

The competition seems to encourage us to predict the short term price changes (i.e. price changes for the next 2hours) to be able to predict when to sell the coin.
We initially invest $10000 and we need to get as much profit as possible. 

## Methods  
In general, predicting every single timestep for the next 120 mins (2hrs) is very difficult to do well. 
Coin or stock price data is said to have unit root, i.e. it is a random walk that is difficult to predict and because of that we will try, as much as possible, to not go down the direction of multistep time series prediction.  

[LSTM version 1](https://github.com/puzzlecollector/bitTrader/blob/main/LSTM_time_price_forecast.ipynb) 
- This model tries to predict the maximum price obtained in the next 2 hours and the time to sell the stock in the next 2 hours. 
- This model ends up $5326 (loss) 

[LSTM version 2](https://github.com/puzzlecollector/bitTrader/blob/main/LSTM_buy_quantity_sell_time.ipynb)
- This model tries to predict the buy quantity and the sell time in the next 2 hours. 
- This model ends up with $7588 (loss)

[Catboost](https://github.com/puzzlecollector/bitTrader/blob/main/Catboost%20test.ipynb) 
- This model tries to predict the price after 2hours for each timestep. 
- It ends up with $6572 (loss)


[GRU N+K Prediction](https://github.com/puzzlecollector/bitTrader/blob/main/GRU%20full%20N%2BK%20Prediction.ipynb)
- This model predicts the price at t+N+K, given t,t+1,...,t+N as input 
- This appears to do better than a single timestep ahead prediction (judging from the plots), but I still believe it will incur a loss upon submission. 
- I have submitted the version that only uses close prices, and it results in $16813 (profit) on public leaderboard. 
- In terms of calculating buy quantity, I am trying to employ a method where the buy quantity is proportional to the number of points that are greater than or equal to the buy price within the next 2 hours after buying. 


