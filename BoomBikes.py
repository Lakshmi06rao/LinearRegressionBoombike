#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import all libraries required for the analysis
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Read the csv file as a dataframe making use of pandas method
bbikeAll = pd.read_csv('C:/Users/hp/Downloads/day.csv')
bbikeAll.isnull()


# In[3]:


#drop any column that had maximum of null data
bbikeAll = bbikeAll.dropna(axis=1, how='all')


# In[4]:


#drop any duplicates if found
bbikeAll.drop_duplicates()

#dteday is not required as the mnth and yr already holds the date value that we require, hence drop the column
def convert_date(date_str):
    return pd.to_datetime(date_str)

bbikeAll['dteday'] = bbikeAll['dteday'].apply(convert_date)
bbikeAll['dates'] = bbikeAll['dteday'].dt.day
bbikeAll


# In[5]:


#display the first 5 records in the dataframe
bbike = bbikeAll.drop('dteday', axis=1)

#drop any other columns that are insignificant for the further process
#here holiday and weekday are already represented in workingday as 0 or 1
bbike = bbike.drop('holiday', axis=1)
bbike = bbike.drop('weekday', axis=1)

bbike.head()


# In[6]:


bbike.info()


# In[7]:


#### numerical variable analysis ####


# In[8]:


sns.pairplot(bbike)
plt.show()


# In[9]:


#analysis for actual numerical predicting variables
sns.pairplot(data=bbike, x_vars=['temp', 'atemp', 'hum', 'windspeed'], y_vars='cnt')


# In[10]:


sns.regplot(x='temp', y='cnt', data=bbike)


# In[11]:


# high/strong positive correlation between cnt and temp/atemp
# weak positve correlation between cnt and hum and windspeed


# In[12]:


bbike.describe()


# In[13]:


#### analysis of categorical variables ####


# In[14]:


weathersit_str_mapping = {
    1: "Clear",
    2: "Mist",
    3: "Light Snow/Rain",
    4: "Heavy Rain"
}

season_str_mapping = {
    1: "spring",
    2: "summer",
    3: "fall",
    4: "winter"
}

bbike['weathersit'] = bbike['weathersit'].map(weathersit_str_mapping)
bbike['season'] = bbike['season'].map(season_str_mapping)


# In[15]:


# box plot to show the usage of boom bikes 

plt.figure(figsize=(20, 9))

#on a working day, annotated as 1
plt.subplot(2,3,1)
sns.boxplot(x='workingday', y='cnt', data=bbike)

#0n years 2018(0) and 2019(1)
plt.subplot(2,3,2)
sns.boxplot(x='yr', y='cnt', data=bbike)

#on days how the weather was
plt.subplot(2,3,3)
sns.boxplot(x='weathersit', y='cnt', data=bbike)

# # on days how the season was
plt.subplot(2,3,4)
sns.boxplot(x='season', y='cnt', data=bbike)


# In[16]:


sns.barplot(x='season', y='cnt', data=bbike)
plt.show()


# In[17]:


# creating dummy variables for weathersit
weather = pd.get_dummies(bbike['weathersit'], drop_first=True)
weather = weather.astype(int)


# In[18]:


bbike = bbike.drop('weathersit', axis=1)
bbike


# In[19]:


#concat dummy varfiable dataframe with the main dataframe
bbike = pd.concat([bbike, weather], axis=1)
bbike


# In[20]:


# creating dummy variables for season
season = pd.get_dummies(bbike['season'], drop_first=True)
season = season.astype(int)
season


# In[21]:


#concat dummy varfiable dataframe with the main dataframe
bbike = pd.concat([bbike, season], axis=1)
bbike


# In[22]:


bbike = bbike.drop('season', axis=1)
bbike


# In[23]:


bbike.corr()


# In[24]:


plt.figure(figsize=(20, 12))
sns.heatmap(bbike.corr(), annot=True)


# In[ ]:





# In[25]:


#splitting data into train and test sets
# df_train, df_test = train_test_split(bbike, train_size=0.7, random_state=100)
# print(df_train.shape)
# print(df_test.shape)


# In[26]:


# min max scaling as per min max scaling(normalisation)
# numericVal = ['temp', 'atemp', 'hum', 'windspeed']
# scaler = MinMaxScaler()
# df_train[numericVal] = scaler.fit_transform(df_train[numericVal])
# df_train


# In[27]:


#training the model


# In[28]:


# Split the data
# Check on variable if it is significant and are correlated to other variables


# In[29]:


X = bbike[['yr', 'workingday', 'temp', 'hum', 'windspeed', 'Light Snow/Rain', 'Mist','spring','summer', 'winter']]
y = bbike['cnt']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=100)
df_train, df_test = train_test_split(bbike, train_size=0.7, random_state=100)

y_train = df_train.pop('cnt')
X_train = df_train[['yr', 'workingday', 'temp', 'hum', 'windspeed', 'Light Snow/Rain', 'Mist','spring','summer', 'winter']]

var = ['yr', 'workingday', 'temp', 'hum', 'windspeed', 'Light Snow/Rain', 'Mist','spring','summer', 'winter']
scaler = MinMaxScaler()
X_train[var] = scaler.fit_transform(X_train[var])
X_train


# In[30]:


plt.figure(figsize=(16, 10))
sns.heatmap(X_train.corr(), annot=True, fmt=".2f")
plt.show()


# In[31]:


#add a contant to check for the future models
X_train_sm = sm.add_constant(X_train['temp'])

#create the first model
first_lr = sm.OLS(y_train, X_train_sm)

first_lr_model = first_lr.fit()

first_lr_model.params


# In[32]:


first_lr_model.summary()


# #The r squared value here is 0.416, meaning 42% of the vraiance was explained by the model containing only temp 

# In[33]:


#add multiple variable
X_train_sm = sm.add_constant(X_train[['temp', 'hum', 'windspeed']])

#create the first model
sec_lr = sm.OLS(y_train, X_train_sm)

sec_lr_model = sec_lr.fit()

sec_lr_model.params


# In[34]:


#summary
sec_lr_model.summary()


# #The r squared value here is 0.477, meaning 47% of the vraiance was explained by the model containing multiple variables 

# In[35]:


#add all variable
X_train_sm = sm.add_constant(X_train[['yr', 'workingday', 'temp', 'hum', 'windspeed', 'Light Snow/Rain', 'Mist','spring','summer', 'winter']])

#create the first model
lr = sm.OLS(y_train, X_train_sm)

lr_model = lr.fit()

lr_model.params


# In[36]:


lr_model.summary()


# In[37]:


#calculating vif
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# Now comparing both the OLS summary and the vif, we can see that hum, temp has high VIF but the p value is low. 
# From OLS we can see that only spring and workingday has high P value greater than 0.05.
# hence, first remove spring and check the VIF values for the other variables

# In[38]:


#remove variable at a time, which has high p value

Xvar = X_train.drop('spring', axis=1)
Xvar = Xvar.drop('Mist', axis=1)


# X = X.drop('workingday', axis=1)
# X = X.drop('summer', axis=1)


# In[39]:


#create new model
X_train_sm_new = sm.add_constant(Xvar)

#create the first model
v1_lr = sm.OLS(y_train, X_train_sm_new)

v1_lr_model = v1_lr.fit()

v1_lr_model.params


# In[40]:


v1_lr_model.summary()


# In[41]:


#calculating vif
vif2 = pd.DataFrame()
vif2['Features'] = Xvar.columns
vif2['VIF'] = [variance_inflation_factor(Xvar.values, i) for i in range(Xvar.shape[1])]
vif2['VIF'] = round(vif2['VIF'], 2)
vif2 = vif2.sort_values(by = 'VIF', ascending = False)
vif2


# In[42]:


#now drop the value that has the highest vif value
Xvar = Xvar.drop('hum', axis=1)


# In[43]:


#create aonther new model
X_train_sm_newest = sm.add_constant(Xvar)

#create the first model
v2_lr = sm.OLS(y_train, X_train_sm_newest)

v2_lr_model = v2_lr.fit()

v2_lr_model.params


# In[44]:


v2_lr_model.summary()


# In[45]:


#calculating vif for the new set
vif3 = pd.DataFrame()
vif3['Features'] = Xvar.columns
vif3['VIF'] = [variance_inflation_factor(Xvar.values, i) for i in range(Xvar.shape[1])]
vif3['VIF'] = round(vif3['VIF'], 2)
vif3 = vif3.sort_values(by = 'VIF', ascending = False)
vif3


# In[46]:


# Residual analysis for the X_train_sm_newest
y_train_pred = v2_lr_model.predict(X_train_sm_newest)
y_train_pred


# In[47]:


residual = y_train - y_train_pred

#distribution of the error 
sns.distplot(residual)


# In[48]:


var = ['yr', 'workingday', 'temp', 'hum', 'windspeed', 'Light Snow/Rain', 'Mist','spring','summer', 'winter']
df_test[var] = scaler.transform(df_test[var])
df_test


# In[49]:


df_test.describe()


# In[50]:


y_test = df_test.pop('cnt')
X_test = df_test[['yr', 'workingday', 'temp', 'hum', 'windspeed', 'Light Snow/Rain', 'Mist','spring','summer', 'winter']]


# In[51]:


#adding constant to the test set
X_test_sm = sm.add_constant(X_test)
X_test_sm.head()


# In[52]:


#drop variables that were previously dropped from X_train set to match the columns
X_test_sm = X_test_sm.drop(['spring', 'Mist', 'hum'], axis=1)


# In[53]:


y_test_pred = v2_lr_model.predict(X_test_sm)
y_test_pred


# In[54]:


# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# mse


# In[55]:


r2 = r2_score(y_true=y_test, y_pred=y_test_pred)
r2


# In[56]:


#The R-squared value of 0.76 suggests that the model has a good fit. 
#This means that 76% of the variability in the target variable can be explained by the independent variables in the model.
#R-squared of 0.76 indicates a robust linear regression model,
#suggesting a solid relationship between the independent and dependent variables.

