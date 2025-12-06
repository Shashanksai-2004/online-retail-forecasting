#!/usr/bin/env python
# coding: utf-8

# ## Importing the Libraries and Loading Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats
get_ipython().system('pip install prophet')
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
get_ipython().system('pip install lightgbm')


# In[2]:


df = pd.read_csv("online_retail.csv")


# ## Data Overview

# In[3]:


df.shape
df.head()
df.info()
df.describe(include='all')


# In[4]:


df.info()
df.isna().sum()


# In[5]:


df.isna().sum().sort_values(ascending=False)


# ## Data Cleaning & Preprocessing

# In[6]:


df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]


# In[7]:


df = df.dropna(subset=['Customer ID'])
df['Customer ID'] = df['Customer ID'].astype(int)


# In[8]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Description'] = df['Description'].astype(str)


# In[9]:


df['Revenue'] = df['Quantity'] * df['Price']


# In[10]:


def clean_retail_data(df):
    df = df.copy()
    
    # Remove invalids
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    df = df.dropna(subset=['Customer ID'])
    
    # Types
    df['Customer ID'] = df['Customer ID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Create metrics
    df['Revenue'] = df['Quantity'] * df['Price']
    
    return df

df_clean = clean_retail_data(df)


# In[11]:


df_clean['zscore'] = (df_clean['Revenue'] - df_clean['Revenue'].mean()) / df_clean['Revenue'].std()
outliers = df_clean[df_clean['zscore'] > 3]
outliers.head()


# In[12]:


Q1 = df_clean['Revenue'].quantile(0.25)
Q3 = df_clean['Revenue'].quantile(0.75)
IQR = Q3 - Q1

outliers_iqr = df_clean[(df_clean['Revenue'] < (Q1 - 1.5 * IQR)) |
                        (df_clean['Revenue'] > (Q3 + 1.5 * IQR))]
outliers_iqr.head()


# ## Data Featuring

# In[13]:


df_clean['date'] = df_clean['InvoiceDate'].dt.date
df_clean['year'] = df_clean['InvoiceDate'].dt.year
df_clean['month'] = df_clean['InvoiceDate'].dt.month
df_clean['weekday'] = df_clean['InvoiceDate'].dt.weekday
df_clean['hour'] = df_clean['InvoiceDate'].dt.hour


# ## Exploratory Data Analysis

# In[14]:


total_rev = df_clean['Revenue'].sum()
num_customers = df_clean['Customer ID'].nunique()
num_products = df_clean['StockCode'].nunique()
num_invoices = df_clean['Invoice'].nunique()

total_rev, num_customers, num_products, num_invoices


# In[15]:


daily = df_clean.groupby('date')['Revenue'].sum()

plt.figure(figsize=(15,4))
plt.plot(daily.index, daily.values)
plt.title("Daily Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()


# In[16]:


monthly = df_clean.resample('M', on='InvoiceDate')['Revenue'].sum()

plt.figure(figsize=(12,4))
monthly.plot()
plt.title("Monthly Revenue Trend")
plt.show()


# In[17]:


country_rev = df_clean.groupby('Country')['Revenue'].sum().sort_values(ascending=False)
country_rev.head(10).plot(kind='bar', figsize=(10,4), title="Top Countries by Revenue")
plt.show()


# In[18]:


top_products = df_clean.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)

top_products.plot(kind='bar', figsize=(14,4), title="Top 10 Products by Revenue")
plt.show()


# In[19]:


pivot = df_clean.pivot_table(values='Revenue', index='weekday', columns='date', aggfunc='sum')

plt.figure(figsize=(16,5))
sns.heatmap(pivot, cmap='YlGnBu')
plt.title("Daily Revenue Calendar Heatmap")
plt.show()


# In[20]:


hour = df_clean.groupby(df_clean['InvoiceDate'].dt.hour)['Revenue'].sum()

plt.figure(figsize=(12,4))
hour.plot(kind='line')
plt.title("Hourly Revenue Trend")
plt.xlabel("Hour of Day")
plt.ylabel("Revenue")
plt.show()


# In[21]:


df_clean['is_weekend'] = df_clean['weekday'].apply(lambda x: 1 if x>=5 else 0)

plt.figure(figsize=(7,4))
sns.boxplot(x='is_weekend', y='Revenue', data=df_clean)
plt.xticks([0,1], ["Weekday", "Weekend"])
plt.title("Revenue Distribution: Weekday vs Weekend")
plt.show()


# In[22]:


plt.figure(figsize=(10,6))
sns.heatmap(df_clean.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Full Numeric Correlation Heatmap")
plt.show()


# In[23]:


daily_df = daily.reset_index()
daily_df['zscore'] = (daily_df['Revenue'] - daily_df['Revenue'].mean()) / daily_df['Revenue'].std()

outliers = daily_df[daily_df['zscore'] > 3]

plt.figure(figsize=(14,4))
plt.plot(daily_df['date'], daily_df['Revenue'])
plt.scatter(outliers['date'], outliers['Revenue'], color='red')
plt.title("Revenue with Outliers Highlighted")
plt.show()


# In[24]:


daily = df_clean.groupby('date')['Revenue'].sum()
daily_ma = daily.rolling(window=30).mean()

plt.figure(figsize=(14,4))
plt.plot(daily, label='Daily Revenue')
plt.plot(daily_ma, label='30-Day Moving Avg', linewidth=3)
plt.title("Revenue Trend with Moving Average")
plt.legend()
plt.show()


# In[25]:


country_rev = df_clean.groupby('Country')['Revenue'].sum().reset_index()

fig = px.choropleth(country_rev, 
                    locations='Country', 
                    locationmode='country names',
                    color='Revenue', 
                    title="Revenue by Country")
fig.show()


# In[26]:


sample_prod = df_clean['Description'].value_counts().index[0]  # top product
prod_daily = df_clean[df_clean['Description'] == sample_prod].groupby('date')['Revenue'].sum()

plt.figure(figsize=(14,4))
prod_daily.plot()
plt.title(f"Daily Revenue Trend for {sample_prod}")
plt.show()


# In[27]:


snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df_clean.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'Revenue': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm.head()


# ## Statistical Analysis

# In[28]:


df_clean[['Quantity','Price','Revenue']].skew()
df_clean[['Quantity','Price','Revenue']].kurt()


# In[29]:


df_clean[['Quantity','Price','Revenue']].cov()
df_clean[['Quantity','Price','Revenue']].corr()


# In[30]:


weekend = df_clean[df_clean['weekday']>=5]['Revenue']
weekday = df_clean[df_clean['weekday']<5]['Revenue']

stats.ttest_ind(weekend, weekday, equal_var=False)


# In[31]:


rev_country = [group['Revenue'].values for name, group in df_clean.groupby('Country')]
stats.f_oneway(*rev_country)


# In[32]:


from scipy.stats import chi2_contingency

table = pd.crosstab(df_clean['Country'], df_clean['StockCode'])
chi2_contingency(table)


# In[33]:


numeric_cols = ['Quantity', 'Price', 'Revenue']
sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.show()


# In[34]:


result = seasonal_decompose(daily, model='additive', period=30)
result.plot()
plt.show()


# In[35]:


sample_prod = df_clean['Description'].value_counts().index[0]  # top product
prod_daily = df_clean[df_clean['Description'] == sample_prod].groupby('date')['Revenue'].sum()

plt.figure(figsize=(14,4))
prod_daily.plot()
plt.title(f"Daily Revenue Trend for {sample_prod}")
plt.show()


# In[36]:


plt.figure(figsize=(10,6))
sns.heatmap(df_clean.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Full Numeric Correlation Heatmap")
plt.show()


# In[37]:


plt.figure(figsize=(15,4))
plt.plot(daily.index, daily.values)
plt.title("Daily Revenue Trend")
plt.savefig("daily_revenue_trend.png")


# In[38]:


summary = {
    "total_revenue": total_rev,
    "unique_customers": num_customers,
    "unique_products": num_products
}

pd.DataFrame([summary]).to_excel("retail_summary.xlsx", index=False)


# In[39]:


df_clean.to_csv("cleaned_online_retail.csv", index=False)


# ## Feature Engineering

# In[40]:


daily = df.groupby(df['InvoiceDate'].dt.date)['Revenue'].sum()
daily = daily.asfreq('D', fill_value=0)


# In[41]:


import numpy as np, pandas as pd

# safe repair + lags
def ensure_daily_with_total_revenue(daily):
    # if Series -> convert
    if isinstance(daily, pd.Series):
        colname = daily.name if daily.name is not None else 'total_revenue'
        daily = daily.to_frame(name=colname)

    # normalize column names
    daily.columns = [c.strip() for c in daily.columns]

    # try to find a revenue-like column
    if 'total_revenue' not in daily.columns:
        possible = [c for c in daily.columns if c.lower() in ('revenue','total_revenue','total','sales','amount')]
        if len(possible) >= 1:
            daily.rename(columns={possible[0]:'total_revenue'}, inplace=True)
        else:
            # fallback: the single numeric column
            numeric_cols = daily.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) == 1:
                daily.rename(columns={numeric_cols[0]:'total_revenue'}, inplace=True)
            else:
                raise ValueError(f"Could not determine revenue column. Columns: {daily.columns.tolist()}")

    # ensure datetime index
    if not np.issubdtype(daily.index.dtype, np.datetime64):
        if 'date' in daily.columns:
            daily['date'] = pd.to_datetime(daily['date'])
            daily = daily.set_index('date').sort_index()
        else:
            try:
                daily.index = pd.to_datetime(daily.index)
            except Exception as e:
                raise ValueError("Index is not datetime and no 'date' column found.") from e

    # create lag features
    daily['lag_1']  = daily['total_revenue'].shift(1)
    daily['lag_7']  = daily['total_revenue'].shift(7)
    daily['lag_30'] = daily['total_revenue'].shift(30)
    return daily

# apply
daily = ensure_daily_with_total_revenue(daily)
daily[['total_revenue','lag_1','lag_7','lag_30']].head()


# In[42]:


daily = df.groupby(df['InvoiceDate'].dt.date)['Revenue'].sum()
daily = daily.asfreq('D', fill_value=0)


# In[43]:


daily = df_clean.groupby(df_clean['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
daily['date'] = pd.to_datetime(daily['InvoiceDate'])
daily = daily.set_index('date').sort_index()
daily = daily[['Revenue']].asfreq('D', fill_value=0)
daily = daily.rename(columns={'Revenue': 'total_revenue'})


# In[44]:


df_feat = daily.copy()
df_feat['dayofweek'] = df_feat.index.dayofweek
df_feat['month'] = df_feat.index.month


# In[45]:


df_feat['lag_1'] = df_feat['total_revenue'].shift(1)
df_feat['lag_7'] = df_feat['total_revenue'].shift(7)
df_feat['lag_30'] = df_feat['total_revenue'].shift(30)


# In[46]:


df_feat['roll_mean_7'] = df_feat['total_revenue'].shift(1).rolling(7).mean()
df_feat['roll_std_7'] = df_feat['total_revenue'].shift(1).rolling(7).std()


# In[47]:


df_feat = df_feat.dropna()


# In[48]:


df_prophet = daily.reset_index().rename(columns={'date':'ds','total_revenue':'y'})


# In[49]:


ts = daily['total_revenue']


# In[50]:


X = df_feat.drop(columns=['total_revenue'])
y = df_feat['total_revenue']


# ## Forecasting Models

# In[51]:


def mape(y_true,y_pred):
    eps = 1e-8
    return np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.array(y_true) + eps))) * 100
def rmse(y_true,y_pred):
    return mean_squared_error(y_true,y_pred, squared=False)

# assume df_clean exists (cleaned transactions with InvoiceDate and Revenue)
# Build daily series and minimal features (as we discussed)
daily = df_clean.groupby(df_clean['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
daily['date'] = pd.to_datetime(daily['InvoiceDate'])
daily = daily.set_index('date').sort_index()
daily = daily[['Revenue']].asfreq('D', fill_value=0)
daily = daily.rename(columns={'Revenue':'total_revenue'})

# minimal feature DataFrame for ML
df_feat = daily.copy()
df_feat['dayofweek'] = df_feat.index.dayofweek
df_feat['month']     = df_feat.index.month
df_feat['lag_1']  = df_feat['total_revenue'].shift(1)
df_feat['lag_7']  = df_feat['total_revenue'].shift(7)
df_feat['lag_30'] = df_feat['total_revenue'].shift(30)
df_feat['roll_mean_7'] = df_feat['total_revenue'].shift(1).rolling(7).mean()
df_feat['roll_std_7']  = df_feat['total_revenue'].shift(1).rolling(7).std()

# drop initial NaNs created by shifts
df_feat = df_feat.dropna()

# Define common train/test split (adjust dates/horizon as needed)
horizon = 30   # forecast horizon (days)
test_end = df_feat.index.max()
test_start = test_end - pd.Timedelta(days=horizon-1)
train_end = test_start - pd.Timedelta(days=1)

print("Train_end:", train_end.date(), "Test range:", test_start.date(), "to", test_end.date())


# In[52]:


prophet_df = daily.reset_index().rename(columns={'date':'ds','total_revenue':'y'})

# Train / test split for Prophet by date
train_prophet = prophet_df[prophet_df['ds'] <= train_end]
test_prophet  = prophet_df[(prophet_df['ds'] >= test_start) & (prophet_df['ds'] <= test_end)]

# Fit Prophet
m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.fit(train_prophet)

# Forecast horizon
future = m.make_future_dataframe(periods=horizon, freq='D')
fcst = m.predict(future)

# Extract predicted values for test period
pred_prophet = fcst.set_index('ds').loc[test_prophet['ds'], 'yhat'].values
actual = test_prophet['y'].values

# Metrics
print("Prophet results:")
print("MAPE: {:.2f}%".format(mape(actual, pred_prophet)))
print("RMSE: {:.2f}".format(rmse(actual, pred_prophet)))
print("MAE:  {:.2f}".format(mean_absolute_error(actual, pred_prophet)))

# Plot
plt.figure(figsize=(12,4))
plt.plot(test_prophet['ds'], actual, label='Actual')
plt.plot(test_prophet['ds'], pred_prophet, label='Prophet Pred', linestyle='--')
plt.title('Prophet: Actual vs Predicted')
plt.legend(); plt.show()


# In[53]:


# SARIMAX model (statsmodels)
# pip install statsmodels

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Use series (daily['total_revenue'])
train_series = daily.loc[:train_end, 'total_revenue']
test_series  = daily.loc[test_start:test_end, 'total_revenue']

# Choose orders — simple starting values (tune for better performance)
order = (1,1,1)
seasonal_order = (0,1,1,7)  # weekly seasonality example

print("Fitting SARIMAX (this may take a moment)...")
try:
    sarima_model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
    sarima_res = sarima_model.fit(disp=False, maxiter=50)
    preds = sarima_res.get_forecast(steps=horizon).predicted_mean
    preds.index = test_series.index  # align indices
except Exception as e:
    print("SARIMAX failed:", e)
    # fallback naive forecast: last observed value repeated
    preds = pd.Series(np.repeat(train_series.iloc[-1], horizon), index=test_series.index)

# Metrics
actual = test_series.values
pred_sarimax = preds.values
print("SARIMAX results:")
print("MAPE: {:.2f}%".format(mape(actual, pred_sarimax)))
print("RMSE: {:.2f}".format(rmse(actual, pred_sarimax)))
print("MAE:  {:.2f}".format(mean_absolute_error(actual, pred_sarimax)))

# Plot
plt.figure(figsize=(12,4))
plt.plot(test_series.index, actual, label='Actual')
plt.plot(test_series.index, pred_sarimax, label='SARIMAX Pred', linestyle='--')
plt.title('SARIMAX: Actual vs Predicted')
plt.legend(); plt.show()


# In[54]:


# LightGBM model (recursive forecasting)

import lightgbm as lgb

# Prepare ML dataset (df_feat prepared earlier)
# X contains the features; y is target
X = df_feat.drop(columns=['total_revenue'])
y = df_feat['total_revenue']

# Ensure train/test split aligns with df_feat index
X_train = X.loc[:train_end]
y_train = y.loc[:train_end]
X_test_full = X.loc[test_start:test_end]   # we'll use dates to drive recursive preds

# Train LightGBM on train set (predict next-day revenue)
lgb_train = lgb.Dataset(X_train, y_train)
params = {'objective':'regression', 'metric':'rmse', 'verbosity':-1, 'seed':42}
gbm = lgb.train(params, lgb_train, num_boost_round=500)

# Recursive forecasting for horizon: feed predictions as inputs for next step
preds = []
# we need a mutable copy of recent history for lags
history = daily['total_revenue'].loc[:train_end].tolist()

test_idx = pd.date_range(start=test_start, end=test_end, freq='D')
for day in test_idx:
    # construct feature row for 'day' using history
    row = {}
    # lags
    row['lag_1'] = history[-1] if len(history) >= 1 else 0
    row['lag_7'] = history[-7] if len(history) >= 7 else history[0]
    row['lag_30'] = history[-30] if len(history) >= 30 else history[0]
    # rolling mean/std for 7
    row['roll_mean_7'] = np.mean(history[-7:]) if len(history) >= 1 else 0
    row['roll_std_7'] = np.std(history[-7:]) if len(history) >= 1 else 0
    # date features
    row['dayofweek'] = day.dayofweek
    row['month'] = day.month

    X_row = pd.DataFrame([row])
    pred = gbm.predict(X_row)[0]
    preds.append(pred)
    history.append(pred)  # append predicted value for next day features

pred_lgbm = np.array(preds)
actual = daily['total_revenue'].loc[test_start:test_end].values

# Metrics
print("LightGBM results:")
print("MAPE: {:.2f}%".format(mape(actual, pred_lgbm)))
print("RMSE: {:.2f}".format(rmse(actual, pred_lgbm)))
print("MAE:  {:.2f}".format(mean_absolute_error(actual, pred_lgbm)))

# Plot
plt.figure(figsize=(12,4))
plt.plot(daily.loc[test_start:test_end].index, actual, label='Actual')
plt.plot(daily.loc[test_start:test_end].index, pred_lgbm, label='LightGBM Pred', linestyle='--')
plt.title('LightGBM (recursive): Actual vs Predicted')
plt.legend(); plt.show()


# In[55]:


X_full = df_feat.drop(columns=['total_revenue'])
y_full = df_feat['total_revenue']

lgb_train_full = lgb.Dataset(X_full, y_full)
params = {'objective':'regression','metric':'rmse','verbosity':-1}
final_model = lgb.train(params, lgb_train_full, num_boost_round=600)


# In[56]:


future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)


# ## Preparing Data for Power BI

# In[57]:


daily.to_csv("daily_revenue.csv")


# In[58]:


forecast.to_csv("forecast_output.csv")


# In[59]:


product_df = df_clean.groupby('Description').agg({
    'Quantity':'sum',
    'Revenue':'sum'
}).reset_index()

product_df.to_csv("product_summary.csv")

