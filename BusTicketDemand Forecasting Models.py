import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"df_selected.csv")


df_new= df[['Date','Tickets Sold']]

#######################  changed datatype because when treating outliers , outliers replaced by float value  ########

df_new.loc[:,'Tickets Sold'] = df_new['Tickets Sold'].astype(int)


######################################      grouping by date column        ###################################


df_grouped = df_new.groupby('Date', as_index=False)['Tickets Sold'].sum()

df_grouped.loc[:, 'Date'] = pd.to_datetime(df_grouped['Date'], format="%Y-%m-%d").dt.date


# Create a complete date range from the minimum to the maximum date in the column
start_date = df_grouped['Date'].min()
end_date = df_grouped['Date'].max()

# Create a complete date range
full_date_range = pd.date_range(start=start_date, end=end_date)

# Find missing dates by comparing the full date range to the dates in the DataFrame
missing_dates = full_date_range.difference(df_grouped['Date'])

missing_dates_df = pd.DataFrame({
    'Date': missing_dates.date,
    'Tickets Sold': [None] * len(missing_dates)  
})

missing_dates_df['Tickets Sold'] = missing_dates_df['Tickets Sold'].astype(float)

df_concatenated = pd.concat([df_grouped, missing_dates_df], axis=0, ignore_index=True)
df_sorted_dates = df_concatenated.sort_values(by='Date', ascending=True).reset_index(drop=True)

df_sorted_dates['Tickets Sold'] = df_sorted_dates['Tickets Sold'].ffill()


####################################################### stationarity ######################################




from statsmodels.tsa.stattools import adfuller

# Perform the Augmented Dickey-Fuller test
result = adfuller(df_sorted_dates['Tickets Sold'])

# Print the ADF test results
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

# Interpreting the result
if result[1] <= 0.05:
    print("The time series is likely stationary.")
else:
    print("The time series is likely non-stationary.")



def mean_absolute_percentage_error(y_true, y_pred): 
    # Convert y_true and y_pred to numpy arrays for ease of computation
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Calculate absolute percentage error for each prediction
    absolute_percentage_error = np.abs((y_true - y_pred) / y_true)
    # Take the mean of all absolute percentage errors and multiply by 100 to get MAPE
    return np.mean(absolute_percentage_error) * 100

###################################### ARIMA ####################################################################

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df3 = df_sorted_dates
df3.set_index('Date', inplace=True)


Train = df3['Tickets Sold'].head(888)  # Selecting the first 147 rows as training data
Test = df3['Tickets Sold'].tail(222)

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(Train, lags=20, ax=plt.gca())  # ACF plot
plt.subplot(122)
plot_pacf(Train, lags=20, ax=plt.gca())  # PACF plot
plt.show()

from statsmodels.tsa.arima.model import ARIMA


Train.plot(figsize=(10, 5), title='Tickets Sold')
plt.show()

model = ARIMA(Train, order=(1,0,1))  # (p,d,q)
model_fit = model.fit()

forecast_steps = len(Test)
forecast1 = model_fit.forecast(steps=forecast_steps)

plt.plot(Test,label='Test Data')
plt.plot(forecast1, color='red')
plt.legend()

mape1 = mean_absolute_percentage_error(Test, forecast1)
mape1


###########################################################  SARIMA #####################################

from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(Train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 365))  # Define seasonal order
fitted_model = model.fit()

forecast_steps = len(Test)
forecast2 = fitted_model.forecast(steps=forecast_steps)

plt.plot(Test,label='Test Data')
plt.plot(forecast2, color='red')
plt.legend()

mape2 = mean_absolute_percentage_error(Test, forecast2)
mape2


##########################################################  ETS #############################################

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
model = SimpleExpSmoothing(Train)  # Example for monthly data
fitted_model = model.fit()

forecast3 = fitted_model.forecast(steps=forecast_steps)

plt.plot(Test,label='Test Data')
plt.plot(forecast3, color='red')
plt.legend()

mape3 = mean_absolute_percentage_error(Test, forecast3)
mape3



#########################################   Random Forest ##################################################
from sklearn.ensemble import RandomForestRegressor

df2=df3.copy()
df2.reset_index(inplace=True)
df2['Date'] = pd.to_datetime(df2['Date'])

df2['year'] = df2['Date'].dt.year
df2['month'] = df2['Date'].dt.month
df2['day'] = df2['Date'].dt.day
df2['weekday'] = df2['Date'].dt.weekday  # Monday=0, Sunday=6

df2 = df2.drop('Date', axis=1)

X = df2.drop('Tickets Sold', axis=1)  
Y = df2['Tickets Sold'] 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# Predict on test data
Y_pred = rf.predict(X_test)

Y_test_indexreset = Y_test.reset_index(drop=True)

plt.plot(Y_test_indexreset,label='Test Data')
plt.plot(Y_pred, color='red')
plt.legend()

mape4 = mean_absolute_percentage_error(Y_test, Y_pred)
mape4

#################################################### XGBRegressor  ###################################

from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the model
model.fit(X_train, Y_train)

# Make predictions on the test set
Forecast5 = model.predict(X_test)

plt.plot(Y_test_indexreset,label='Test Data')
plt.plot(Forecast5, color='red')
plt.legend()

mape5 = mean_absolute_percentage_error(Y_test, Forecast5)
mape5

################################################## saving random forest by using pickle #######################

import pickle
pickle.dump(rf, open('rf.pkl', 'wb'))






