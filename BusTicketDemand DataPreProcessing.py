
import pandas as pd
import sweetviz  
import seaborn as sns
import numpy as np
from feature_engine.outliers.winsorizer import Winsorizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt



df = pd.read_csv(r"Operational_Bus_data - Operational_Bus_data.csv")


df.info()
df.describe()

my_report = sweetviz.analyze([df, "df"])
my_report.show_html('Report.html')


#################################         Duplication       #######################################


duplicate = df.duplicated()
sum(duplicate)

#################################        MissingValues      ########################################


df.isna().sum()
#df.dropna(inplace = True)

# Replace missing values in the specified columns with the median
df['Frequency (mins)'] = df['Frequency (mins)'].fillna(df['Frequency (mins)'].median())
df['Distance Travelled (km)'] = df['Distance Travelled (km)'].fillna(df['Distance Travelled (km)'].median())
df['Time (mins)'] = df['Time (mins)'].fillna(df['Time (mins)'].median())

df['Bus Route No.'] = df['Bus Route No.'].fillna('Unknown')





#################################         Typecasting       #######################################

df['Frequency (mins)'] = df['Frequency (mins)'].astype(int)
df['Time (mins)'] = df['Time (mins)'].astype(int)

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y').dt.date


#################################         Outlier Treatment       #######################################



columns_selected = ['Trips per Day','Bus Stops Covered','Frequency (mins)','Distance Travelled (km)','Time (mins)','Tickets Sold','Revenue Generated (INR)']

df[columns_selected].plot(kind='box', subplots=True, layout=(3,5), sharex=False, sharey=False,figsize=(25,18))


winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Trips per Day','Bus Stops Covered','Frequency (mins)','Distance Travelled (km)','Tickets Sold','Revenue Generated (INR)'])

df_t = winsorizer.fit_transform(df[['Trips per Day','Bus Stops Covered','Frequency (mins)','Distance Travelled (km)','Tickets Sold','Revenue Generated (INR)']])

columns_selected2 = ['Trips per Day','Bus Stops Covered','Frequency (mins)','Distance Travelled (km)','Tickets Sold','Revenue Generated (INR)']

df_t[columns_selected2].plot(kind='box', subplots=True, layout=(3,5), sharex=False, sharey=False,figsize=(25,18))

df_drop = df.drop(columns=columns_selected2)

df_selected = pd.concat([df_drop,df_t], axis=1)



################################ replacing values #############################################

df_selected['Way'] = df_selected['Way'].replace({'One-way': 'One Way', 'Round-trip': 'Round Trip'})




###############################    Zero-Variance ##########################     


numeric_columns = df_selected.select_dtypes(include=np.number)
numeric_columns.var()
numeric_columns.var() == 0 
numeric_columns.var(axis=0) == 0 

####################################################### mean ###################################

mean_values = df_selected[['Time (mins)',
       'Trips per Day', 'Bus Stops Covered',
       'Frequency (mins)', 'Distance Travelled (km)',
       'Tickets Sold', 'Revenue Generated (INR)']].mean()
mean_values

####################################################### median ###################################

median_values = df_selected[['Time (mins)',
       'Trips per Day', 'Bus Stops Covered',
       'Frequency (mins)', 'Distance Travelled (km)',
       'Tickets Sold', 'Revenue Generated (INR)']].median()
median_values

####################################################### mode ###################################


mode_values = df_selected[['Time (mins)',
       'Trips per Day', 'Bus Stops Covered',
       'Frequency (mins)', 'Distance Travelled (km)',
       'Tickets Sold', 'Revenue Generated (INR)']].mode()
mode_values

####################################################### standard deviation ###################################


std_values = df_selected[['Time (mins)',
       'Trips per Day', 'Bus Stops Covered',
       'Frequency (mins)', 'Distance Travelled (km)',
       'Tickets Sold', 'Revenue Generated (INR)']].std()
std_values



#####################################################    skewness  #########################################

skewness_values = df_selected[['Time (mins)',
       'Trips per Day', 'Bus Stops Covered',
       'Frequency (mins)', 'Distance Travelled (km)',
       'Tickets Sold', 'Revenue Generated (INR)']].skew()
skewness_values

#####################################################    Kurtosis  #########################################

Kurtosis_values = df_selected[['Time (mins)',
       'Trips per Day', 'Bus Stops Covered',
       'Frequency (mins)', 'Distance Travelled (km)',
       'Tickets Sold', 'Revenue Generated (INR)']].kurtosis()
Kurtosis_values





###################################### univariate plots #####################################################


columns_to_plot = ['Time (mins)',
       'Trips per Day', 'Bus Stops Covered',
       'Frequency (mins)', 'Distance Travelled (km)',
       'Tickets Sold', 'Revenue Generated (INR)']



df_selected[columns_to_plot].hist(bins=20,figsize=(15, 10), alpha=0.7, grid = False)

df_selected[columns_to_plot].plot(kind='box', subplots=True, layout=(3,5), sharex=False, sharey=False,figsize=(15,8))
plt.show()

import scipy.stats as stats

stats.probplot(df_selected['Time (mins)'], dist="norm", plot=plt)
stats.probplot(df_selected['Trips per Day'], dist="norm", plot=plt)
stats.probplot(df_selected['Bus Stops Covered'], dist="norm", plot=plt)
stats.probplot(df_selected['Frequency (mins)'], dist="norm", plot=plt)
stats.probplot(df_selected['Distance Travelled (km)'], dist="norm", plot=plt)
stats.probplot(df_selected['Tickets Sold'], dist="norm", plot=plt)
stats.probplot(df_selected['Revenue Generated (INR)'], dist="norm", plot=plt)


#################################################  multi variate plots #####################################################

dfheat = df_selected[columns_to_plot]
dfheatcr = dfheat.corr()
sns.heatmap(dfheatcr, annot=True, cmap='YlGnBu', linewidths=0.5)
plt.show()


sns.pairplot(dfheat, diag_kind="hist")

####################################### exporting dataset    #################################

df_selected.to_csv('df_selected.csv',index=False)


