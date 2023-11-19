#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf


# In[2]:


sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")


# In[3]:


sp500


# In[4]:


sp500.plot.line(y="Close", use_index=True)


# In[5]:


# Clean up data
columns_to_remove = ["Dividends", "Stock Splits"]
sp500 = sp500.drop(columns=columns_to_remove)
sp500["Tomorrow"] = sp500["Close"].shift(-1)

# Shows whether the the price will increase or decrease
sp500["Trend"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Remove historical data where fundamental shifts in the market might affect prediction
sp500 = sp500.loc["1990-01-01":].copy()

# Remove todays data so we can train the model properly
sp500 = sp500.iloc[:-1]

sp500


# In[6]:


from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier


# In[7]:


# Setting up Random Forest Classifier
features = ["Open", "High", "Low", "Close", "Volume"]

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Create a Random Forest Classifier
randForest = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

randForest.fit(train[features], train["Trend"])


# In[11]:


from sklearn.metrics import precision_score
import pandas as pd
prediction = randForest.predict(test[features])
prediction = pd.Series(prediction, index=test.index)
precision_score(test["Trend"], prediction)


# In[ ]:





# In[ ]:




