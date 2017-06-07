
# coding: utf-8

# In[1]:

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
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import  cross_val_score


# ### Read the metadata

# In[2]:

df_metadata = pd.read_csv('../ebu3b/data/ebu3b_metadata.csv')

def get_df_metadata():
    return df_metadata


# ### 2. Weather data
# 
# - Weather data for La Jolla is purchased from https://openweathermap.org/
# ![weather description](weather_metadata.png)
# 
# **2a. Weather data cleanup**
# 
# - Remove columns that have all / most nans
# - Remove columns that are not applicable (city_id, weather_icon)
# 
# columns that are retained are:
# 
# 1) dt_iso  
# 2) temp  
# 3) temp_min  
# 4) temp_max  
# 5) pressure  
# 6) humidity  
# 7) wind_speed  
# 8) wind_deg  
# 9) clouds_all  
# 10) weather_id  
# 11) weather_main
# 
# **2b. Feature engineering**
# 
# 1) Conversion of temperature columns from Klevin to farenheit  
# 2) Label encoding for weather_main

# ### Read the weather data

# In[3]:

# Columns required from weather data
req_columns = ['dt_iso', 'temp', 'temp_min', 'temp_max', 'pressure', 'humidity',
               'wind_speed', 'wind_deg', 'clouds_all', 'weather_id', 'weather_main']
# Read the weather data frame
weather_df = pd.read_csv('../ebu3b/weather.csv')[req_columns]

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

def get_df_weather():
    return weather_df


# ### Get the dataframe for the given signals
# 
# - Takes the room, list of signals, mean_type, weather as input
# - if mean_type is hour it groups by hour and averages the value for each hour
# - If mean_type is quarter_hout it groups by every 15 minutes
# - If weather is True it combines with weather data or returns the raw data frame.

# In[4]:

data_path = "../ebu3b/data/"

# Returns the bucket time based on original minutes and bucket_size
def get_bucket(min_hour, bucket_size):
    return (min_hour / bucket_size).astype("int") * bucket_size

def get_signal_dataframe(room, signals = None, mean_type="hour", use_weather_data=True):
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

        min_bucket_size = None
        if mean_type == "hour":
            min_bucket_size = 60
        elif mean_type == "quarter_hour":
            min_bucket_size = 15
        elif mean_type == '5minutes':
            min_bucket_size = 5

        hour_bucket_size = None
        if mean_type == 'day':
            hour_bucket_size = 24
        elif mean_type == 'quarter_day':
            hour_bucket_size = 6

        if not min_bucket_size is None:
            hours = df.time.dt.hour.astype("str") + " hours"
            mins = get_bucket(df.time.dt.minute, min_bucket_size).astype("str") + " minutes"
            # Groupby hour and average the values per hour. This is to reduce the number of data points.
            groupby_item = pd.to_datetime(df.time.dt.date) + pd.to_timedelta(hours + " " + mins)

        if not hour_bucket_size is None:
            hours = get_bucket(df.time.dt.hour, hour_bucket_size).astype("str") + " hours"
            # Groupby hour and average the values per hour. This is to reduce the number of data points.
            groupby_item = pd.to_datetime(df.time.dt.date) + pd.to_timedelta(hours)
        
        if not mean_type is None:
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
    rm_signals = df_all.pivot_table(values='value', index=['time', 'location'],                                                   columns="Ground Truth Point Type").reset_index()
    if use_weather_data:
        rm_signals = weather_df.merge(rm_signals, left_on='dt_iso', right_on='time')
    return rm_signals


# ### Model for the day
# 
# - Takes input the data frame and the day for which model must be running
# - It runs the linear regression, Lasso, Ridge, DecisionTreeRegressor, AdaBoostRegressor on the data

# In[16]:

def model_for_day(model_df, features, target, day='Sunday'):
    model_df = model_df.dropna()
    X = model_df[model_df.dt_iso.dt.weekday_name == day][features]
    y = model_df[model_df.dt_iso.dt.weekday_name == day][target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    list_df = []
    reg = DecisionTreeRegressor().fit(X_train, y_train)
    cv_scorestrain = cross_val_score(reg, X_train, y_train, cv=10)
    cv_scorestest = cross_val_score(reg, X_test, y_test, cv=10)
    list_df.append(pd.DataFrame({'model' : 'DecisionTreeRegressor', 'train score' : cv_scorestrain.mean(),
                  'test score' : cv_scorestest.mean()}, index=[0]))

    reg = LinearRegression().fit(X_train, y_train)
    cv_scorestrain = cross_val_score(reg, X_train, y_train, cv=10)
    cv_scorestest = cross_val_score(reg, X_test, y_test, cv=10)
    list_df.append(pd.DataFrame({'model' : 'LinearRegression', 'train score' : cv_scorestrain.max(),
                  'test score' : cv_scorestest.mean()}, index=[0]))

    reg = Lasso().fit(X_train, y_train)
    cv_scorestrain = cross_val_score(reg, X_train, y_train, cv=10)
    cv_scorestest = cross_val_score(reg, X_test, y_test, cv=10)
    list_df.append(pd.DataFrame({'model' : 'Lasso', 'train score' : cv_scorestrain.mean(),
                  'test score' : cv_scorestest.mean()}, index=[0]))

    reg = Ridge().fit(X_train, y_train)
    cv_scorestrain = cross_val_score(reg, X_train, y_train, cv=10)
    cv_scorestest = cross_val_score(reg, X_test, y_test, cv=10)
    list_df.append(pd.DataFrame({'model' : 'Ridge', 'train score' : cv_scorestrain.mean(),
                  'test score' : cv_scorestest.mean()}, index=[0]))

    reg = AdaBoostRegressor().fit(X_train, y_train)
    cv_scorestrain = cross_val_score(reg, X_train, y_train, cv=10)
    cv_scorestest = cross_val_score(reg, X_test, y_test, cv=10)
    list_df.append(pd.DataFrame({'model' : 'AdaBoostRegressor', 'train score' : cv_scorestrain.mean(),
                  'test score' : cv_scorestest.mean()}, index=[0]))

    return pd.concat(list_df, ignore_index=True)


# ### API: getTime and plotResults
# 
# - plotResults is the utility function that will plot the compressed and reconstructed data.

# In[6]:

def getTime(x, dfTest):
    return dfTest.at[int(x),'timeseries']

def plotResults(dfs, plotTemplates, method, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_title('compression analysis for ' + method)
    linestyles = ['_', '-', '--', ':']
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    timeX=dfs[0]['timeseries'].tolist()
    axes = [ax, ax.twinx()]
    axes1Count=0
    axes0Count=0
    for i in xrange(len(plotTemplates)):
        try:
            ReconDF=dfs[2*i+1].to_frame(name='values')
            ReconDF['time']=ReconDF.index
            ReconDF=ReconDF.dropna()
            ReconDF=ReconDF.sort_values(by=['time'], ascending=[1])
            ReconDF['timeseries']=ReconDF.apply(lambda x: getTime(x['time'],dfs[2*i]), axis=1)
            if(plotTemplates[i] in ['Actual Supply Flow','Occupied Command','Damper Position']):
                axes1Count=1
                axes[1].plot(dfs[2*i]['timeseries'].tolist(),dfs[2*i]['values'].tolist(),
                             'k--',color=colors[i],label=plotTemplates[i])
                axes[1].plot(ReconDF['timeseries'].tolist(),ReconDF['values'].tolist(),
                             'k:',color=colors[i],label=plotTemplates[i]+'_reconstructed')
            else:
                axes0Count=1
                axes[0].plot(dfs[2*i]['timeseries'].tolist(),dfs[2*i]['values'].tolist(),
                             'k--',color=colors[i],label=plotTemplates[i])
                axes[0].plot(ReconDF['timeseries'].tolist(),ReconDF['values'].tolist(),
                             'k:',color=colors[i],label=plotTemplates[i]+'_reconstructed')
        except:
            print "Exception for : ", plotTemplates[i]
    if(axes1Count==0):
        axes[1].plot(timeX,[0 for x in timeX],'k',color='w',label='axes1')
        axes[1].grid(True)
    if(axes0Count==0):
        axes[0].plot(timeX,[0 for x in timeX],'k',color='w',label='axes0')
        axes[0].grid(True)
    axes[0].legend(loc='upper left',fontsize='x-small')
    axes[1].legend(loc='upper right',fontsize='x-small')
    axes[1].set_ylabel('scale for top right legend')
    axes[0].set_ylabel('scale for top left legend')


# ### CLASS: encoder
# 
# **encoder is the base class for the piecewise approiximation model.**
# 
# 1. This is the base class for the piecewise approximation models.
# 2. **compress()** will compress the data.
# 3. **recon()** will reconstruct the compressed data.
# 4. **compute_error()** will compute the error.

# In[7]:

class encoder:
    """
    The encoder/decoder class is the base class for all encoder/decoder pairs.
    Subclasses encode different types of encoding.
    EncoderLearner is a factory class for fitting encoders to data
    """
    def __init__(self, raw, max_gap):
        """
        given a spark DataFrame or Series (raw), find the best model of a given type
        """

    def compress(self):
        """
        given a raw sequence and a model, return a compressed representation.
        """
        self.compressed=None
        return self.compressed
    
    def recon(self,compressed):
        """
        Recreate the original DataFrame or Series, possibly with errors.
        """
        Recon=None
        return Recon
    
    def get_size(self):
        return len(self.compressed)
    
    def compute_error(self,S,compressed):
        if type(compressed)==type(None):
            compressed=self.compressed
        #R=self.recon(compressed=compressed,index=S.index)
        R=self.recon()
        V=R-S
        V=V.dropna()
        return np.sqrt(sum([v*v for v in V.values]))/len(V)


# ### CLASS: piecewise_constant
# 
# **piecewise_linear is the class that performs the piecewise constant approximation on the data. **
# 
# 1. This class inherits encoder class.
# 2. compress and recon are overridden from base class.
# 3. internal method fit() is used to fit the data.
# 
# 
# **Constructor:**
# 
# - Takes input as series.
# - Calls fit() to train the model.
# 
# **fit**
# 
# - Takes series of values and max_gap as input.
# - Uses dynamic programming to create patches of minimized error.
# - Error is calculated iteratively by finding the error and number of switches.
# - values for minimum error is stored in each iteration.
# 
# **recon**
# 
# - Takes the array of time, value pairs and create a treashold points.
# - NaNs are removed and are not interpolated to have constant values.
# 
# **Compress**
# 
# - Creates an array of {time, value} values based on the current and previous value.

# In[8]:

class piecewise_constant(encoder):
    """Represent the signal using a sequence of piecewise constant functions 
    """

    def __init__(self, S, max_gap):
        if type(S) != pd.Series:
            raise 'encode expects pandas Series as input'
        # Save the index and call the fit to update the model parameters.
        self.index = S.index;
        self.Sol = self.fit(S, max_gap)
        print "piecewise_constant model initialized"
    
    def fit(self, S, max_gap):
        print "Fitting piecewise_constant"
        # Replace the nan values with 0
        S.fillna(0)
        
        # Calculate the range of values.
        # _range is a constant that is added to the error at each stop point
        # Larger values will cause fewer switches.
        _range=np.max(S) - np.min(S)
        print 'range = ', _range

        # Dynamic programming.
        # An array that holds the best partition ending at each point of the sequence.
        # Each element contains a best current value, a pointer to the last change in best
        # solution so far and the total error of best solution so far.
        Sol = [[]] * len(S)
        for i in range(len(S)):
            if i == 0:
                Sol[i] = {'prev' : None, 'value' : S[0], 'error' : 0.0, 'switch_no' : 0}
            # Sol is indexed by the location in the sequence S
            # prev: the index of the last switch point
            # value: current prediction value
            # error: cumulative error to this point
            # switch_no: number of switches so far.
            else:
                # Calculate the squared error with previous value.
                err0 = Sol[i-1]['error'] + (Sol[i-1]['value'] - S[i]) ** 2
                best, best_err, best_val = None, 1e20, S[i]
                for j in xrange(np.max([0, i - max_gap]), i):
                    
                    # Calculate the mean and standard deviation of gap.
                    _mean, _std = np.mean(S[j : i]), np.std(S[j : i])
                    
                    # Calculate the error
                    err = _std * (i - j) + Sol[j]['error'] + _range
                    
                    # Compare and get the best params
                    if err < best_err:
                        best, best_val, best_err = j, _mean, err
                # Store the best params.
                Sol[i] = {'prev' : best, 'value' : best_val, 'error' : best_err,                        'switch_no': Sol[best]['switch_no'] + 1}
        return Sol

    def compress(self, S):
        """Compress the data."""
        # Initiallize the switch points.
        Switch_points = []

        # start from the end 
        i = len(self.Sol) - 1

        while i > 0:
            prev, value = self.Sol[i]['prev'], self.Sol[i]['value']
            if self.Sol[prev]['value'] != value:
                Switch_points.append({'time':S.index[prev],'value':value})
            i = prev
        self.compressed = Switch_points
        return Switch_points

    def recon(self, compressed = None, index = None):
        """Reconstructs the data from compressed data
        """
        if type(index) == type(None):
            index = self.index
        Recon = pd.Series(index=index)

        if type(compressed) == type(None):
            compressed = self.compressed
        for e in compressed:
            time = e['time']
            Recon[time] = e['value']

        return Recon.fillna(method = 'ffill')


# ### CLASS: piecewise_linear
# 
# **piecewise_linear is the class that performs the piecewise linear approximation on the data. **
# 
# 1. This class inherits encoder class.
# 2. compress and recon are overridden from base class.
# 3. internal method fit() is used to fit the data.
# 
# 
# **Constructor:**
# - Takes input as series.
# - Calls fit() to train the model.
# 
# **fit**
# - Takes series of values and max_gap as input.
# - Uses dynamic programming to create patches of minimized error.
# - Error is calculated iteratively by finding the slope for each max_gap.
# - values for minimum error is stored in each iteration.
# 
# **recon**
# 
# - Linear interpolation is performed for reconstruction from the compresseed data.
# - numpy interpolation function is used.
# 
# **Compress**
# 
# - Iteratively stores the time and values for every changing slope

# In[9]:

class piecewise_linear(encoder):
    """ 
    Represent the signal using a sequence of piecewise linear functions 
    """
    def __init__(self, S, max_gap):
        if type(S) != pd.Series:
            raise 'encode expects pandas Series as input'
        # Save the index and call the fit to update the model parameters.
        self.index = S.index
        self.Sol = self.fit(S, max_gap)
        print "piecewise_linear model initialized"
    
    # fit uses dynamic programming to find the best piecewise linear solution
    # max_gap is the maximal extent of a single step.
    # Reason for max_gap is that even if the error is small we want to correct
    # it with some minimal frequence. 
    # Not quite a snapshot because the value will not necessarily change after 
    # max_gap is reached.
    def fit(self, S, max_gap):
        print "Fitting piecewise_linear"
        # Replace the nan values with 0
        S.fillna(0)
        
        # Calculate the range of values.
        # _range is a constant that is added to the error at each stop point
        # Larger values will cause fewer switches.
        _range=np.max(S) - np.min(S)
        print 'range = ', _range

        # Dynamic programming.
        # An array that holds the best partition ending at each point of the sequence.
        # Each element contains a best current value, a pointer to the last change in best
        # solution so far and the total error of best solution so far.
        Sol = [[]] * len(S)
        for i in range(len(S)):
            if i == 0:
                Sol[i]={'prev':None, 'value':S[0], 'error':0.0, 'switch_no':0, 'slope':0}
            # Sol is indexed by the location in the sequence S
            # prev: the index of the last switch point
            # value: current prediction value
            # error: cumulative error to this point
            # slope: slope of th linear line at this point
            else:
                err0 = Sol[i-1]['error'] + (Sol[i-1]['value'] - S[i]) ** 2
                best, best_err, best_val, best_slope = None, 1e20, S[i], 1e20
                for j in xrange(np.max([0, i - max_gap]), i):
                    
                    # Calculate the slope
                    _slope=(S[i] - S[j]) * 1.0 / (i - j)
                    
                    # Initialize the parameters.
                    _val, _err = 0, 0
                    for k in xrange(j, i):
                        # Calculate the new value based on slope.
                        _val = Sol[j]['value'] + _slope * (k - j)
                        
                        # Calculate the error.
                        _err += (Sol[k]['value'] - _val) ** 2
                    
                    # Calculate the total error.
                    # Need to understand why _range is addeed to error.
                    err = _err * 1.0 / (i - j) + Sol[j]['error'] + _range
                    _val = Sol[j]['value'] + _slope * (i - j)
                    
                    # Compare and get the best params.
                    if err < best_err:
                        best, best_val, best_err, best_slope, = j, _val,err,_slope

                # Save the best params
                Sol[i] = {'prev':best, 'value':best_val, 'error':best_err,                        'switch_no': Sol[best]['switch_no']+1, 'slope':best_slope}

        # Return the fit parameters.
        return Sol

    def compress(self,S):
        Switch_points = []
        
        # start from the end 
        i = len(self.Sol) - 1
        while i > 0:
            prev, slope, value, = self.Sol[i]['prev'], self.Sol[i]['slope'], self.Sol[i]['value']
            if self.Sol[prev]['slope'] != slope:
                Switch_points.append({'time' : S.index[prev], 'value' : value})
            i = prev

        # Save the compressed data and return the data.
        self.compressed = Switch_points
        return Switch_points

    def recon(self,compressed=None, index=None):
        if type(index)==type(None):
            index = self.index

        # Initialize the recon series.
        Recon = pd.Series(index=index)

        if type(compressed) == type(None):
            compressed = self.compressed
        for e in compressed:
            time = e['time']
            Recon[time] = e['value']
        
        # Interpolate the value using linear method.
        Recon.interpolate(method="linear", inplace=True)
        return Recon


# ### API: model
# 
# **model is the API that calls piecewise linear / piecewise constant based on method,
# compresses, recostructs and calculates the error.**
# 
# 1. Get the encoder object by calling piecewise_constant / piecewise_linear
# 2. Compress the data.
# 3. Reconstruct the data.
# 4. Calculate the compression error.
# 5. Returns the compressed and reconstructed data frames.

# In[10]:

def model(pd_df, method, tolerance):
    """model calls either piecewise_constant or piecewise_linear based on the method.
    It gets the appropriate encoder, compresses, reconstructs the data and calculates the error.
    """

    # Get the values
    S = pd_df['values']
    
    # Calculate the standard deviation of the values.
    _std = np.std(S)
    print "Std dev is ", _std
    # Call piecewise_constant / piecewise_linear API based on method and get the encoder.
    if method == 'piecewise_constant':
        encoder = piecewise_constant(S, tolerance)
    elif(method == 'piecewise_linear'):
        encoder = piecewise_linear(S, tolerance) # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX chage it to linear
    
    # Compress the data as per the encoder.
    C = encoder.compress(S)
    
    # Reconstruct the encoded data.
    R = encoder.recon()

    # Create the data frame of comparessed data.
    compressed_df = pd.DataFrame(C)

    # Calculate the error between compressed and original data.
    error = encoder.compute_error(S, compressed=C)
    
    print 'error =', error, 'error/_std=',error/_std
    
    # Return the compressed and re constructed dataframes.
    return [compressed_df, R]


# ### API: runAnalysis
# 
# **runAnalysis is the API that performs piecewise linear / piecewise constant analysis on a given teamplate.**
# 
# 1. Create a query based on template.
# 2. Project timeseries and values.
# 3. Convert the data into appropriate datatypes.
# 4. Call the model on the data.
# 5. Returns the compressed and reconstructed data frames.

# In[11]:

def runAnalysis(room, stTime = None, enTime = None, templates = ['Zone Temperature'], method='piecewise_linear'):
    """runAnalysis is the API that performs piecewise linear / piecewise constant analysis on a given teamplate."""
    
    # Initialize the data frames and templates to plot.
    dfs = []
    plotTemplates=[]
    dfs_compressed = []
    
    # Run the analysis for each template upto the template count.
    for t in templates:
        try:
            # get the signal dataframe for specified room and signals
            if type(room) is pd.core.frame.DataFrame:
                dataDF = room
            elif type(room) is str:
                dataDF = get_signal_dataframe(room, [t], mean_type=None, use_weather_data=False)

            if not stTime is None:
                stTime = pd.to_datetime(stTime)
                dataDF = dataDF[dataDF.time >= stTime]
            if not enTime is None:
                enTime = pd.to_datetime(enTime)
                dataDF = dataDF[dataDF.time <= enTime]

            if len(dataDF) < 50:
                print "small data frame. length = ", len(dataDF)

            # rename the index in sequence
            dataDF.index = range(len(dataDF))

            # Rename the columns names from time, <template> to timesries to values
            dataDF = dataDF[['time', t]].rename(columns={'time' : 'timeseries', t : 'values'})

            # Run the model and get the compressed dataframe and reconstructed data frame.
            [compressedDF, reconDF] = model(dataDF, method, tolerance = 10)
            
            # Append the template and output data frames of model.
            plotTemplates.append(t)
            dfs.extend([dataDF, reconDF])
            dfs_compressed.append(compressedDF)
        except:
            print "Exception for template: ", t

    # Return the templates and dataframes.
    return [dfs, plotTemplates, dfs_compressed]


# ### API: CompressWithPCA
# 
# **CompressWithPCA is the API that performs PCA on the data frame and plots the original and reconstructed for template.**
# 
# 1. Takes the data frame as input
# 2. Performs PCA, transforms and reconstructs the data frame with the number of components specified
# 3. filters the data frame based on date range
# 4. selects the required template from original and reconstructed
# 5. returns the data frames

# In[12]:

def CompressWithPCA(dataDF, stTime, enTime, template='Zone Temperature', n_components = 9):
    # Fill the nan values with its previous values
    dataDF.fillna(method='bfill', inplace = True)

    # Remove first 2 columns. first 2 colunms are date and location.
    original_df = dataDF.iloc[:, 2:]

    pca = PCA(n_components = n_components).fit(original_df)
    transformed = pca.transform(original_df)
    reconstructed = pca.inverse_transform(transformed)
    reconstructed_df = pd.concat([dataDF.iloc[:,0:2], pd.DataFrame(reconstructed, columns=dataDF.columns[2:])], axis=1)

    if not stTime is None:
        stTime = pd.to_datetime(stTime)
        dataDF = dataDF[(dataDF.time >= stTime)]
    if not enTime is None:
        enTime = pd.to_datetime(enTime)
        dataDF = dataDF[(dataDF.time <= enTime)]
    
    df_orig = dataDF.rename(columns={'time':'timeseries', template:'values'})
    df_reconstruct = reconstructed_df[(reconstructed_df.time >= stTime) & (reconstructed_df.time <= enTime)][template]
    return [[df_orig, df_reconstruct], [template], [transformed]]


# ### API: run_length_encoding
# 
# **run_length_encoding is the API that performs run length encoding on the signals specified.**
# 
# 1. Inputs: room / dataframe, signals to compress, stTime, enTime, tolerance, plot template
# 2. Performs run length encoding on the signals
# 3. filters the data frame based on date range
# 4. returns the original, compressed and reconstructed dataframes, data frames.

# In[13]:

def same_bucket(se1, se2, tolerance):
    mask = (se1 - se2).abs() > tolerance
    val = False if mask.sum() else True
    return val

def run_length_encoding(room, signals = None, stTime = None, enTime = None, tolerance = None, template='Zone Temperature'):
    
    # get the signal dataframe for specified room and signals
    if type(room) is pd.core.frame.DataFrame:
        dataDF = room
    elif type(room) is str:
        dataDF = get_signal_dataframe(room, signals, mean_type=None, use_weather_data=False)

    # filter the data for the required time
    if not stTime is None:
        stTime = pd.to_datetime(stTime)
        dataDF = dataDF[dataDF.time >= stTime]
    if not enTime is None:
        enTime = pd.to_datetime(enTime)
        dataDF = dataDF[dataDF.time <= enTime]

    # rename the index in sequence
    dataDF.index = range(len(dataDF))

    # calculate the tolerance
    if tolerance is None:
        tolerance = [pd.to_timedelta("15 minutes")] + [0] * (len(dataDF.columns) - 1)

    # initial reference is the first row
    count, num_rows = 1, len(dataDF)
    reference1, reference2 = dataDF.iloc[0, :].copy(), dataDF.iloc[0, :].copy()
    
    # initialize the empry compresed data frame
    compressed_df = pd.DataFrame(columns=dataDF.columns.tolist() + ["count"])
    
    # run the loop for every row
    for i in range(1, num_rows):
        se = dataDF.iloc[i, :]
        if not same_bucket(reference2, se, tolerance):
            reference1['count'] = count
            compressed_df = compressed_df.append(reference1)
            reference1, reference2 = se.copy(), se.copy()
            count = 1
        else:
            count += 1
            reference2['time'] = se['time']
    reference1['count'] = count
    compressed_df = compressed_df.append(reference1)

    recon_df = pd.DataFrame(dataDF.time)
    recon_df = recon_df.merge(compressed_df, how='left').fillna(method='ffill')

    df_orig = dataDF.rename(columns={'time':'timeseries', template:'values'})
    return [[df_orig, recon_df[template]], [template], [compressed_df]]


# ## K Mean Clustering on the Room Data
# 
# **What**
# - Perform KMeans on the data to group them into various clusters.
# 
# **Why**
# - Compression using PCA on clusters is lot better then the whole data.
# 
# **Procedure**
# 1. Define kmeans to provide the cluster and its centers
# 2. Calculate RMS Error based on the Predictions and evaluate the clustering Effectiveness
# 3. Evaluate the right k value plotting the RMS error on a elbow curve
# 4. Based on the elbow curve below, k = 4 is the most ideal
# 5. Now using k=4 Run the k means cluster again and determine the cluster each row in the dataframe belongs to
# 6. Using Each Cluster , now run them through PCA for 1 of the signals "Zone Temperature" to understand the patterns on each cluster

# In[14]:

def performKmeans(n_clusters, dataDF):
    # Initialize the kmeans
    kmeans = KMeans(n_clusters=n_clusters)

    # classify the data in to clusters
    predicted = kmeans.fit_predict(dataDF)
    
    return predicted, kmeans.cluster_centers_

def calculateRMSE(original_df, labels, centers):
    labels_df = pd.DataFrame(labels, columns=['cluster'])
    clustered_df = labels_df.merge(pd.DataFrame(centers), left_on='cluster', right_index=True, how='left')
    return mean_squared_error(original_df, clustered_df.iloc[:, 1:])   


# In[ ]:



