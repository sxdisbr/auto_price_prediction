#!/usr/bin/env python
# coding: utf-8

# # Car cost prediction

# A company of used cars is developing an application to attract new customers. In it, you can quickly find out the market value of your car. Historical data is at our disposal: technical specifications, complete sets and prices of cars. We need to build a model to predict the cost. 
# 
# Important to the customer:
# 
# - prediction quality;
# - prediction speed;
# - training time.

# ## Data preproccesing

# In[1]:


import sys

# Ensuring pip is up to date
get_ipython().system('{sys.executable} -m pip install --upgrade pip')

# Installing CatBoost using pip with the --user flag
get_ipython().system('{sys.executable} -m pip install catboost --user')

# Installing lightgbm using pip with the --user flag
get_ipython().system('{sys.executable} -m pip install lightgbm --user')



# In[2]:


get_ipython().system('pip install scikit-learn==1.1.3')
import re
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from catboost import CatBoostRegressor
from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer


# In[3]:


data=pd.read_csv(r'C:\Users\pinos\Downloads\autos.csv')


# In[4]:


data.head()


# In[5]:


display(data.dtypes)


# There are many string types in the dataset, which suggests that our predictive analysis task will be a regressive classification task, since our goal is the int type.

# Looking for duplicates.

# In[65]:


data.duplicated().sum()/len(data)*100


# The number of duplicated files is ridiculous, so we deleted them.

# In[7]:


data = data.drop_duplicates()


# Now looking for missing values.

# In[8]:


data.isna().mean()


# Here the amount is quite significant, especially in relation to the repaired, but also in relation to the type of vehicle, and somewhat less in relation to the type of fuel and gearbox.

# First, we are going to visualize those variables that will help in forecasting.

# In[9]:


len(data['Power'])**0.5/2


# In[10]:


plt.figure(figsize=(20,10))
plt.hist(data['Power'], bins=30, range=(0, 400))
plt.show()


# We are amazed by the number of frequencies that we find with zero power, which leads us to believe that this is an error that needs to be fixed by removing them from the dataset. 
# In addition, both below 50 and above 300 are atypical values that we can exclude.

# In[11]:


data = data.query('Power > 50' and 'Power < 300')


# In[12]:


plt.figure(figsize=(20,10))
plt.hist(data['Power'], bins=40, range=(50, 400))
plt.show()


# We perform the same operation for the Price variable.

# In[13]:


plt.figure(figsize=(20,10))
plt.hist(data['Price'], bins=30, color='green')
plt.show()


# In[14]:


data[data['Price'] < 100]['Price'].count()


# Below 100 euros, we have almost 7,481 cars, since it is unlikely that a car costs less than 100 euros, we will delete this data.

# In[15]:


data=data.query('Price > 100')


# In[16]:


plt.figure(figsize=(20,10))
plt.hist(data['Price'], bins=30, color='green')
plt.show()


# We are checking the registration date variable.

# In[17]:


print(pd.to_datetime(data['DateCrawled'].min()))
print(pd.to_datetime(data['DateCrawled'].max()))


# In[18]:


print(pd.to_datetime(data['RegistrationYear'].min()))
print(pd.to_datetime(data['RegistrationYear'].max()))


# We filter by the appropriate time parameters.

# In[19]:


data = data.query('1970 < RegistrationYear < 2017')


# In[20]:


plt.figure(figsize=(20,10))
plt.hist(data['RegistrationYear'], bins=100, color='red')
plt.show()


# We are going to fill the remaining nan with an unknown category.

# In[21]:


data[['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Repaired']]=data[
    ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Repaired']].fillna('unknown')


# In[22]:


data.isna().mean()


# In[23]:


plt.figure(figsize=(20, 10))
h=sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True, center=True)
h.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=16)


# As we can see in the correlation matrix, the most related variables are Power, Kilometers with Price. these are the variables we dealt with when debugging data.

# ### Conclusions
# In this first section of our research, we got acquainted with our dataset and sorted out the missing data by choosing the best solution: delete the data that we considered an error and fill in the categorical variables with the category unknown.

# ## Training model

# We move on to preparing and splitting the data before applying training and forecasting models.

# We have removed the Postal Code column as it is of no value.

# In[24]:


data=data.drop(['PostalCode', 'DateCrawled', 'RegistrationMonth', 
                'LastSeen', 'DateCreated', 'NumberOfPictures','RegistrationYear'], axis=1)


# In[25]:


data.columns


# In[26]:


data=data.reset_index(drop=True)


# In[27]:


data.head()


# We have divided our data set into training and test.

# In[28]:


target=data['Price']
features=data.drop(['Price'], axis=1)


# We perform the function of visualizing real data and forecasts.

# In[73]:


def graphics(test,predict):
    MSE = np.square(np.subtract(y_test, predict)).mean() 
    RMSE= math.sqrt(MSE)
    plt.figure(figsize=(20, 10))
    pd.Series(predict).hist(bins=20, alpha=0.7)
    pd.Series(test).hist(bins=20, alpha=0.7)
    plt.text(16122,11225,'MAE={:.2f}'.format(MSE))
    plt.text(16122,10225,'RMSE={:.2f}'.format( RMSE))
    plt.legend(['Predicted values', 'Real values'])
    plt.title('A graph of the distribution of predicted and true values', y=1.05)
    plt.xlabel('Predicted / Real')
    plt.ylabel('Quantity')
    plt.subplot()   
    plt.figure(figsize=(20, 10))
    plt.plot(predict,test,'o', alpha=0.8)
    plt.xlim(-5000,25000)
    plt.ylim(-5000,25000)
    plt.plot([-10000,25000],[-10000,25000],'--y', linewidth=2)
    plt.title('Graph of the ratio of predicted values to true values', y=1.05)
    plt.xlabel('Predicted values')
    plt.ylabel('Real values')
    plt.show()


# We process categorical and numerical ones to then create a pipeline that will allow us to train and then predict.

# In[30]:


numeric_features = ['Power', 'Kilometer']


# In[31]:


numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


# In[32]:


categorical_features = ['VehicleType', 'Gearbox', 'FuelType', 'Brand', 'Repaired']


# In[33]:


categorical_transformer_forest = Pipeline(
    steps=[
        ("encoder", OrdinalEncoder(handle_unknown="error", dtype=np.int8))
      
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer_forest, categorical_features),
    ]
)


# In[34]:


clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(max_depth=2, random_state=42,
                                                                        
                                                                               n_estimators=50))]
)


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size = 0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# In[36]:


param_grid = {
    "regressor__n_estimators": range(10, 60, 10),
    "regressor__max_depth": [None] + [i for i in range(2,11)]
    
}

search_cv = RandomizedSearchCV(clf, param_grid, n_iter=10, random_state=0)
search_cv


# In[37]:


clf.get_params().keys()


# In[66]:


search_cv.fit(X_train, y_train)

print("Best parameters:")
print(search_cv.best_params_)


# In[67]:


print(f"CV score: {search_cv.best_score_:.3f}")


# In[68]:


print(
    "Accuracy of the best model as a result of randomized search:"
    f"{search_cv.score(X_test, y_test):.3f}"
)


# In[41]:


clf.fit(X_train, y_train)


# In[69]:


print("Model evaluation: %.3f" % clf.score(X_val, y_val))


# As we can see, the model does not give a good result.

# In[43]:


pred_rfr=clf.predict(X_val)


# In[44]:


get_ipython().run_cell_magic('time', '', 'MSE = np.square(np.subtract(y_val, pred_rfr)).mean() \nRMSE_rfr = math.sqrt(MSE) \nprint("RMSE_RandomForestRegressor: ", RMSE_rfr)\n')


# In[45]:


random_forest_regressor = [3531, '2.84 ms', '1.75 ms']


# In[74]:


graphics(y_val,pred_rfr)


# In[47]:


categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(drop='first', handle_unknown="ignore", sparse=False, dtype=np.int8))
      
    ]
)


# In[48]:


clf_lgbm = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(random_state=6))]
)


# In[49]:


lgbm_reg_model = clf_lgbm


# In[50]:


clf_lgbm.fit(X_train, y_train)


# In[70]:


print("Model evaluation: %.3f" % clf_lgbm.score(X_val, y_val))


# The LGBM model gives twice the best result than the previous one.

# In[52]:


pred_lgbm=clf_lgbm.predict(X_val)


# In[53]:


get_ipython().run_cell_magic('time', '', 'MSE = np.square(np.subtract(y_val, pred_lgbm)).mean() \nRMSE_lgbm = math.sqrt(MSE) \nprint("RMSE_LGBM: ", RMSE_lgbm)\n')


# In[54]:


LGBM = [2370, '15.7 ms', '1.55 ms']


# Over time, this also seems like a good option.

# In[75]:


graphics(y_val,pred_lgbm)


# In[56]:


clf_cb = Pipeline(
    steps=[("preprocessor", preprocessor), 
           ("regressor", CatBoostRegressor(random_state=3,silent=True))]
)


# In[57]:


clf_cb.fit(X_train, y_train)


# In[71]:


print("Model evaluation: %.3f" % clf_cb.score(X_val, y_val))


# Cut Boost still shows better results than LGBM, although the difference is not very big.

# In[59]:


pred_cb=clf_cb.predict(X_val)


# In[60]:


get_ipython().run_cell_magic('time', '', 'MSE = np.square(np.subtract(y_val, pred_cb)).mean() \nRMSE_CB = math.sqrt(MSE) \nprint("RMSE_CB: ", RMSE_CB)\n')


# In[61]:


cat_boost = [2253, '2.42 ms', '2.37 ms']


# At the speed level, CatBoost also wins by a small margin.

# In[76]:


graphics(y_val,pred_cb)


# ## Model analysis

# In[72]:


analysis = pd.DataFrame([random_forest_regressor, LGBM, cat_boost], 
             columns=['RMSE training sample', 'Training time', 'Prediction time'], 
                      index=['Random Forest Regressor', 'LGBM', 'Cat Boost'])    

analysis.sort_values(by='RMSE training sample', ascending=True)


# ## Testing the best model

# The best model turned out to be the Cat Boost model, so that's what we're going to test at the final stage of testing. 

# In[64]:


get_ipython().run_cell_magic('time', '', 'pred_cb=clf_cb.predict(X_test)\nMSE = np.square(np.subtract(y_test, pred_cb)).mean() \nRMSE_CB = math.sqrt(MSE) \nprint("RMSE_CB: ", RMSE_CB)\n')


# As we can see, the RMSE is very similar to the one obtained at the validation stage, and the time is slightly longer than it was obtained at the validation stage, but even in this case we can call it fast.

# ### Conclusions
# As we can see from the final output, the most recommended model for predicting the price according to the customer's request-the speed of deployment - is CatBoost.
# Both CatBoost and LGBM are better regression solutions than the RandomForestRegressor model, these gradient boosting models show results twice as good.
# At the predictive level, the tree model also does not work well, since both the LGBM model and the CatBoost model, the predictive results are closer to the real results, as we see in the visualization graphs. Nevertheless, as a winner in the field of forecasting, LGBM performs better than CatBoost, since its results are closer to real data. As for speed, the clear winner is also the CatBoost model.
