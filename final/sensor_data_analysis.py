import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor

df_metadata = pd.read_csv('ebu3b/data/ebu3b_metadata.csv')

# Columns required from weather data
req_columns = ['dt_iso', 'temp', 'temp_min', 'temp_max', 'pressure', 'humidity',
               'wind_speed', 'wind_deg', 'clouds_all', 'weather_id', 'weather_main']
# Read the weather data frame
weather_df = pd.read_csv('ebu3b/weather.csv')[req_columns]

# Remove UTC and convert to datetime
weather_df['dt_iso'] = pd.to_datetime(weather_df.dt_iso.str.replace(" UTC", ""))
# Convert to pacific standard time.
weather_df = weather_df.set_index('dt_iso').tz_localize('UTC').tz_convert('US/Pacific').reset_index()

# Strip time zone info and convert to datetime.
weather_df['dt_iso'] = pd.to_datetime(weather_df.dt_iso.astype('str').str.replace("-07:00", ""))

# Convert temperatures to farenheit
weather_df['temp'] = weather_df.temp * 9 / 5 - 459.67
weather_df['temp_min'] = weather_df.temp_min * 9 / 5 - 459.67
weather_df['temp_max'] = weather_df.temp_max * 9 / 5 - 459.67

# Encode the weather_main
weather_df['weather_main'] = LabelEncoder().fit_transform(weather_df.weather_main)

# Create datetime columns
weather_df['date'] = weather_df.dt_iso.dt.date
weather_df['hour'] = weather_df.dt_iso.dt.hour

def get_df_metadata():
    return df_metadata

def get_df_weather():
    return weather_df

data_path = "./ebu3b/data/"
def get_signal_dataframe(room, signals = None, hour_mean=True):
    df_list = []
    df_filtered = df_metadata[(df_metadata.Location == room)]
    if not signals is None:
        df_filtered = df_filtered[df_filtered['Ground Truth Point Type'].isin(signals)]

    for identifier in df_filtered['Unique Identifier'].values:
        # Filename is same as identifier
        filename = identifier + ".csv"
        
        # Read the csv 
        df = pd.read_csv(data_path + filename).dropna()
        
        # Convert to datetime object
        df["time"] = pd.to_datetime(df.time)

        if hour_mean:
            # Groupby to average the
            groupby_item = pd.to_datetime(df.time.dt.date) + pd.to_timedelta(df.time.dt.hour.astype("str") + " hours")
            df = df.groupby(groupby_item)[['value']].mean().reset_index()

        # Create new colunms
        df["identifier"], df['location'] = identifier, room

        # append the data frame to list
        df_list.append(df)
        
        print "Read file: ", filename

    df_all = pd.concat(df_list)

    # Merge the dataframe with meta data and filter hte required columns
    df_all = df_all.merge(df_metadata, right_on="Unique Identifier", left_on="identifier")[["time",
        "value", "identifier", "location", "Ground Truth Point Type"]]
    rm_signals = df_all.pivot_table(values='value', index=['time', 'location'], \
                                                  columns="Ground Truth Point Type").reset_index()
    return weather_df.merge(rm_signals, left_on='dt_iso', right_on='time')

def model_for_day(model_df, features, target, day='Sunday'):
    model_df = model_df.dropna()
    X = model_df[model_df.dt_iso.dt.weekday_name == day][features]
    y = model_df[model_df.dt_iso.dt.weekday_name == day][target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    list_df = []
    reg = DecisionTreeRegressor().fit(X_train, y_train)
    list_df.append(pd.DataFrame({'model' : 'DecisionTreeRegressor', 'train score' : reg.score(X_train, y_train),
                  'test score' : reg.score(X_test, y_test)}, index=[0]))

    reg = LinearRegression().fit(X_train, y_train)
    list_df.append(pd.DataFrame({'model' : 'LinearRegression', 'train score' : reg.score(X_train, y_train),
                  'test score' : reg.score(X_test, y_test)}, index=[0]))

    reg = Lasso().fit(X_train, y_train)
    list_df.append(pd.DataFrame({'model' : 'Lasso', 'train score' : reg.score(X_train, y_train),
                  'test score' : reg.score(X_test, y_test)}, index=[0]))

    reg = Ridge().fit(X_train, y_train)
    list_df.append(pd.DataFrame({'model' : 'Ridge', 'train score' : reg.score(X_train, y_train),
                  'test score' : reg.score(X_test, y_test)}, index=[0]))

    reg = AdaBoostRegressor().fit(X_train, y_train)
    list_df.append(pd.DataFrame({'model' : 'AdaBoostRegressor', 'train score' : reg.score(X_train, y_train),
                  'test score' : reg.score(X_test, y_test)}, index=[0]))

    return pd.concat(list_df, ignore_index=True)