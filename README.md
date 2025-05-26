The  data set contains these columns:-
'Date', 'Bus Route No.', 'From','To','Trips per Day','Way','Bus Stops Covered','Frequency (mins)','Distance Travelled (km)','Time (mins)','Main Station','Tickets Sold','Revenue Generated (INR)'.


Handled missing values using imputation techniques (median, forward fill) and treated outliers using the IQR-based Winsorization method.

Conducted exploratory data analysis (EDA) using statistical measures (mean, median, skewness, kurtosis) and visualized patterns using histograms, box plots, KDEs, pair plots, and correlation heatmaps.

Built interactive dashboards in Power BI to display route-wise and station-wise ticket sales, frequency, and revenue trends.

Performed time series forecasting on daily ticket sales after ensuring stationarity (ADF test passed). Filled missing dates and applied forward-fill strategy.

Trained and evaluated five models: ARIMA, SARIMA, ETS, Random Forest, and XGBRegressor.

Achieved best performance using Random Forest with a MAPE score of 31%.
