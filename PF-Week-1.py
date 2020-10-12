# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from IPython import get_ipython
import sys
get_ipython().system('{sys.executable} -m pip install plotly')
# %%
import sys
!{sys.executable} -m pip install plotly

#%%
!{sys.executable} -m pip install matplotlib
# %%
import numpy as np
import pandas as pd
#%%
import plotly.express as px
import plotly.graph_objects as go
#%%
import matplotlib as plt

# %%
np.random.normal(0, 1, 50)

# %%
np.random.normal(0,1,100)

# %%
!{sys.executable} -m pip install scipy
# %%
from scipy.stats import norm
import random

# %%
random.uniform(0,1)

# %%
for i in range(100):
    print(random.uniform(0,1))


# %%
random_numbers = (np.random.randint(0,10000,size=(10000))/10000)
random_numbers


# %%
#PDF of Uniform Distribution
fig = go.Figure(data = [go.Histogram(x = random_numbers, cumulative_enabled = False)])
fig.update_layout(template = 'plotly_dark')
fig.show()
# %%
#CDF of Uniform Distribution
fig = go.Figure(data = [go.Histogram(x = random_numbers, cumulative_enabled = True)])
fig.update_layout(template = 'plotly_dark')
fig.show()



#%%
random_numbers_normal = np.random.normal(0, 1, 10000)
# %%
#PDF of Normal Distribution
fig = go.Figure(data = [go.Histogram(x = random_numbers_normal, cumulative_enabled = False)])
fig.update_layout(template = 'plotly_dark')
fig.show()
# %%
#CDF of Normal Distribution
fig = go.Figure(data = [go.Histogram(x = random_numbers_normal, cumulative_enabled = True)])
fig.update_layout(template = 'plotly_dark')
fig.show()

#%%
random_numbers_lognormal = np.random.lognormal(0, 1, 10000)
# %%
#PDF of Log_Normal Distribution
fig = go.Figure(data = [go.Histogram(x = random_numbers_lognormal, cumulative_enabled = False)])
fig.update_layout(template = 'plotly_dark')
fig.show()
# %%
#CDF of Log_Normal Distribution
fig = go.Figure(data = [go.Histogram(x = random_numbers_lognormal, cumulative_enabled = True)])
fig.update_layout(template = 'plotly_dark')
fig.show()

#%%
random_numbers_poisson = np.random.poisson(10, 10000)
# %%
fig = go.Figure(data = [go.Histogram(x = random_numbers_poisson, cumulative_enabled = False)])
fig.update_layout(template = 'plotly_dark')
fig.show()

# %%
fig = go.Figure(data = [go.Histogram(x = random_numbers_poisson, cumulative_enabled = True)])
fig.update_layout(template = 'plotly_dark')
fig.show()

# %%
#Linear Regression
random_numbers

#%%
normal_dataframe = pd.DataFrame({'X':random_numbers})
#normal_dataframe["e"] = norm.rvs(0 ,1, 10000)
normal_dataframe["e"] = norm.rvs(0 ,normal_dataframe['X'], 10000)
normal_dataframe["Y"] = normal_dataframe["e"] + 5*normal_dataframe['X']
fig = go.Figure(data = go.Scatter(x = normal_dataframe['X'], y = normal_dataframe["Y"], mode = 'markers'))
fig.update_layout(template = 'plotly_dark')
fig.show()

# %%
!{sys.executable} -m pip install statsmodels

# %%
import statsmodels.api as sm

# Note the difference in argument order
model = sm.OLS(normal_dataframe['Y'], normal_dataframe['X']).fit()
# Print out the statistics
model.summary()

# %%
!{sys.executable} -m pip install sklearn 
# %%
from sklearn.neighbors import KNeighborsClassifier


# %%
# Linear Reg
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# %%

normal_dataframe['X2'] = (np.random.randint(-100000,10000,size=(10000))/10000)
normal_dataframe['Y2'] = normal_dataframe['Y'] + 3*normal_dataframe['X2'] + 5*normal_dataframe['e']
X = normal_dataframe[['X','X2']].values

Y = normal_dataframe['Y2'].values
Y = Y.reshape(-1,1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=10)

# Create the regressor: reg_all
reg_all = LinearRegression().fit(X_train, y_train)

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error: {}".format(rmse))

# %%
#Cross Validation

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, Y, cv  = 15)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 15-Fold CV Score: {}".format(np.mean(cv_scores)))



# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as plt
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
import statsmodels.api as sm
import plotly.figure_factory as ff

# %%
#lets import some financial data
ASX_200 = pd.read_csv("^AXJO.csv")
ASX_200['Date'] =  pd.to_datetime(ASX_200['Date'], format='%Y-%m-%d')
ASX_200 = ASX_200.set_index('Date')
ASX_200.head()



# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=ASX_200.index, y=ASX_200['Adj Close'],
                    mode='lines',
                    name='lines'))
fig.update_layout(template="plotly_dark")

fig.show()


# %%
ANZ = pd.read_csv("ANZ.AX.csv")
ANZ['Date'] =  pd.to_datetime(ANZ['Date'], format='%Y-%m-%d')
ANZ = ANZ.set_index('Date')
ANZ.head()


# %%
CBA = pd.read_csv("CBA.AX.csv")
CBA['Date'] = pd.to_datetime(CBA['Date'], format = '%Y-%m-%d')
CBA = CBA.set_index('Date')
CBA.head()


# %%
NAB = pd.read_csv("NAB.AX.csv")
NAB['Date'] = pd.to_datetime(NAB['Date'], format = '%Y-%m-%d')
NAB = NAB.set_index('Date')
NAB.head()


# %%
WBC = pd.read_csv("WBC.AX.csv")
WBC['Date'] = pd.to_datetime(WBC['Date'], format = '%Y-%m-%d')
WBC = WBC.set_index('Date')
WBC.head()


# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=ANZ.index, y=ANZ['Adj Close'],
                    mode='lines',
                    name='ANZ'))
fig.add_trace(go.Scatter(x = CBA.index, y = CBA['Adj Close'],
                        mode = 'lines',
                        name = 'CBA'))
fig.add_trace(go.Scatter(x=NAB.index, y=NAB['Adj Close'],
                    mode='lines',
                    name='NAB'))
fig.add_trace(go.Scatter(x = WBC.index, y = WBC['Adj Close'],
                        mode = 'lines',
                        name = 'WBC'))
fig.update_layout(template="plotly_dark")
fig.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=ANZ.index, y=ANZ['Adj Close']/ANZ['Adj Close'][0],
                    mode='lines',
                    name='ANZ'))
fig.add_trace(go.Scatter(x = CBA.index, y = CBA['Adj Close']/CBA['Adj Close'][0],
                        mode = 'lines',
                        name = 'CBA'))
fig.add_trace(go.Scatter(x=NAB.index, y=NAB['Adj Close']/NAB['Adj Close'][0],
                    mode='lines',
                    name='NAB'))
fig.add_trace(go.Scatter(x = WBC.index, y = WBC['Adj Close']/WBC['Adj Close'][0],
                        mode = 'lines',
                        name = 'WBC'))
fig.update_layout(template="plotly_dark")
fig.show()
# %%
ANZ_Returns = np.log(ANZ['Adj Close']).diff()
CBA_Returns = np.log(CBA['Adj Close']).diff()
NAB_Returns = np.log(NAB['Adj Close']).diff()
WBC_Returns = np.log(WBC['Adj Close']).diff()
Index_Returns = np.log(ASX_200['Adj Close']).diff()

returns_df = pd.concat([Index_Returns, ANZ_Returns, CBA_Returns, NAB_Returns, WBC_Returns ], axis = 1)
returns_df = returns_df.dropna()
returns_df.columns = ('Index', 'ANZ', 'CBA', 'NAB', 'WBC')
returns_df.head()

#%%
prices_df = pd.concat([ASX_200['Adj Close'],ANZ['Adj Close'], CBA['Adj Close'], NAB['Adj Close'], WBC['Adj Close'] ], axis = 1)
prices_df = prices_df.dropna()
prices_df.columns = ('Index', 'ANZ', 'CBA', 'NAB', 'WBC')
prices_df.head()



# %%
quantile_scores = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
returns_df.describe(quantile_scores)


# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x = returns_df.index, y = returns_df['Index'],
                        mode = 'lines',
                        name = 'Index'))
fig.add_trace(go.Scatter(x = returns_df.index, y = returns_df['ANZ'],
                        mode = 'lines',
                        name = 'ANZ'))
fig.add_trace(go.Scatter(x = returns_df.index, y = returns_df['CBA'],
                        mode = 'lines',
                        name = 'CBA'))
fig.add_trace(go.Scatter(x = returns_df.index, y = returns_df['NAB'],
                        mode = 'lines',
                        name = 'NAB'))
fig.add_trace(go.Scatter(x = returns_df.index, y = returns_df['WBC'],
                        mode = 'lines',
                        name = 'WBC'))
fig.update_layout(title='ASX200 and Big 4 Bank Returns',
                   xaxis_title='Date',
                   yaxis_title='Log Returns')
fig.update_layout(template="plotly_dark")
fig.show()



#%%

from statsmodels.tsa.stattools import adfuller
for bank in ('ANZ', 'WBC', 'NAB', 'CBA'):
    adf_test = adfuller(prices_df[bank])
    print('ADF Statistic: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])
    print('Critical Values:')
    for key, value in adf_test[4].items():
        print(bank + '\t%s: %.3f' % (key, value))



#%%
for bank in ('ANZ', 'WBC', 'NAB', 'CBA'):
    adf_test = adfuller(returns_df[bank])
    print('ADF Statistic: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])
    print('Critical Values:')
    for key, value in adf_test[4].items():
        print(bank + '\t%s: %.3f' % (key, value))
# %%
hist_data = [returns_df['ANZ'], returns_df['NAB'], returns_df['WBC'], returns_df['CBA']]
group_labels = ['ANZ', 'NAB', 'WBC', 'CBA'] # name of the dataset
fig = ff.create_distplot(hist_data, group_labels, bin_size=.001,
                         curve_type='normal',show_rug=True)
fig.update_xaxes(range=[-0.1, 0.1])
fig.update(layout_title_text= 'Bank Returns with Gaussian Distribution')
fig.update_layout(template="plotly_dark")
fig.show()

# %%
hist_data = [returns_df['ANZ'], returns_df['NAB'], returns_df['WBC'], returns_df['CBA']]
group_labels = ['ANZ', 'NAB', 'WBC', 'CBA'] # name of the dataset
fig = ff.create_distplot(hist_data, group_labels, bin_size=.001,
                         curve_type='kde',show_rug=True)
fig.update_xaxes(range=[-0.1, 0.1])
fig.update(layout_title_text= 'Kernel Density of Bank Returns')
fig.update_layout(template="plotly_dark")
fig.show()

#%%
import plotly.express as px
fig = px.scatter_matrix(returns_df)
fig.update_layout(template="plotly_dark")
fig.show()

# %%
returns_df.corr()




# %%
list_short_trades = []
list_long_trades = []
dates = returns_df.iloc[1:-30].reset_index()['Date']
for i in range (0, len(returns_df)-31):

    stock_long = returns_df[['ANZ', 'CBA', 'NAB', 'WBC']].iloc[0+i:30+i].sum().sort_values().idxmin()
    stock_short = returns_df[['ANZ', 'CBA', 'NAB', 'WBC']].iloc[0+i:30+i].sum().sort_values().idxmax()

    short_trade = 0
    long_trade = 0

    short_trade = -1*returns_df[stock_short].iloc[i+31]
    
    long_trade = 1*returns_df[stock_long].iloc[i+31]
    
    list_short_trades.append(short_trade)
    list_long_trades.append(long_trade)
    
 

# %%
trade_returns = pd.concat([dates, pd.DataFrame(list_long_trades), pd.DataFrame(list_short_trades)], axis = 1)
trade_returns.columns = ('Date', 'Long', 'Short')
trade_returns['Total Return'] = trade_returns['Long'] + trade_returns['Short']
trade_returns.describe()
# %%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df['Index'], name="ASX 200",
                    line_shape='linear'))
fig.add_trace(go.Scatter(x=trade_returns['Date'], y=trade_returns['Long'], name="Long",
                    line_shape='linear'))
fig.add_trace(go.Scatter(x=trade_returns['Date'], y=trade_returns['Short'], name="Short",
                    line_shape='linear'))
fig.add_trace(go.Scatter(x=trade_returns['Date'], y=trade_returns['Total Return'], name="Long-Short",
                    line_shape='linear'))                   
fig.update_layout(title = 'Daily Returns')
fig.update_layout(template="plotly_dark")
fig.show()

# %%
import plotly.figure_factory as ff
import numpy as np

# Add histogram data
x1 = trade_returns['Long']
x2 = trade_returns['Total Return']

# Group data together
hist_data = [x1, x2]

group_labels = ['Long', 'Long-Short']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.001, show_rug = True)
fig.update_layout(title = 'Histogram and PDF of Returns')
fig.update_layout(template="plotly_dark")
fig.show()

# %%
import plotly.graph_objects as go
import math
fig = go.Figure()
fig.add_trace(go.Scatter(x=trade_returns['Date'], y= (trade_returns['Long'].cumsum()*10000 + 10000), name="Long",
                    line_shape='linear'))
fig.add_trace(go.Scatter(x=trade_returns['Date'], y= (trade_returns['Short'].cumsum()*10000 + 10000), name="Short",
                    line_shape='linear'))
fig.add_trace(go.Scatter(x=trade_returns['Date'], y= (trade_returns['Total Return'].cumsum()*10000 + 10000), name="Long-Short",
                    line_shape='linear'))
fig.add_trace(go.Scatter(x=returns_df.index, y= (returns_df['Index'].cumsum()*10000 + 10000), name="ASX",
                    line_shape='linear'))
fig.update_layout(title = 'Cumulative Returns with 10k')
fig.update_layout(template="plotly_dark")
fig.show()


# %%
asx_rets = returns_df.reset_index().iloc[31:,1].reset_index()
asx_rets
# %%
trade_returns['ASX 200'] = asx_rets.iloc[:,1]
trade_returns['Excess Return'] = trade_returns['Total Return'] - trade_returns['ASX 200']
trade_returns


#%%
import plotly.express as px
fig = px.scatter(x=trade_returns['ASX 200'], y=trade_returns['Total Return'])
fig.update_layout(template="plotly_dark")
fig.update_layout(
    title="Trading vs ASX 200",
    xaxis_title="Index Returns",
    yaxis_title="Trading",
    
)
fig.show()
# %%
import statsmodels.api as sm # import statsmodels 
y = trade_returns['Total Return']
X = trade_returns['ASX 200']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model



        # Note the difference in argument order
model = sm.GLS(y.astype(float), X.astype(float)).fit() ## sm.GLS(output, input)
        # Print out the statistics
model.summary()


#%%
beta = []
param = []
for i in range(0,499):
    ran = random.randint(0,1000)
    trades_dataframe_regression = trade_returns.iloc[ran:ran+262,:]
    y = trades_dataframe_regression['Total Return']
    X = trades_dataframe_regression['ASX 200']
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model



        # Note the difference in argument order
    model = sm.GLS(y.astype(float), X.astype(float)).fit() ## sm.GLS(output, input)
        # Print out the statistics
    beta.append(model.params['ASX 200'])
    param.append(model.params['const'])

# %%
bootstrap_capm_dataframe = pd.concat([pd.DataFrame(beta), pd.DataFrame(param)], axis = 1)
bootstrap_capm_dataframe.columns = ('Beta', 'Alpha')

bootstrap_capm_dataframe.describe()
#%%



fig = go.Figure()
fig.add_trace(go.Histogram(x=bootstrap_capm_dataframe['Alpha'], name = 'Long-Short Returns',xbins=dict(start=-0.1,
        end=0.0015,
        size=0.0001
    )))
# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.8)
fig.update_layout(title = 'Bootstrapped Alpha')
fig.update_layout(template = 'plotly_dark')
fig.show()

# %%


# %%


# %%
