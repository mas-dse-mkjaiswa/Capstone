
# coding: utf-8

# In[8]:

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sensor_data_analysis import *


# ## API: getTime and plotResults
# 
# - plotResults is the utility function that will plot the compressed and reconstructed data.

# In[9]:

def getTime(x, dfTest):
    return dfTest.at[int(x),'timeseries']

def plotResults(dfs, plotTemplates):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_title('compression analysis')
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


# ## CLASS: encoder
# 
# **encoder is the base class for the piecewise approiximation model.**
# 
# 1. This is the base class for the piecewise approximation models.
# 2. **compress()** will compress the data.
# 3. **recon()** will reconstruct the compressed data.
# 4. **compute_error()** will compute the error.

# In[10]:

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


# ## CLASS: piecewise_constant
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

# In[11]:

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


# ## CLASS: piecewise_linear
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

# In[12]:

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


# ## API: model
# 
# **model is the API that calls piecewise linear / piecewise constant based on method,
# compresses, recostructs and calculates the error.**
# 
# 1. Get the encoder object by calling piecewise_constant / piecewise_linear
# 2. Compress the data.
# 3. Reconstruct the data.
# 4. Calculate the compression error.
# 5. Returns the compressed and reconstructed data frames.

# In[13]:

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


# ## API: runAnalysis
# 
# **runAnalysis is the API that performs piecewise linear / piecewise constant analysis on a given teamplate.**
# 
# 1. Create a query based on template.
# 2. Project timeseries and values.
# 3. Convert the data into appropriate datatypes.
# 4. Call the model on the data.
# 5. Returns the compressed and reconstructed data frames.

# In[14]:

def runAnalysis(room, stTime = None, enTime = None, templates = ['Zone Temperature'], method='piecewise_linear'):
    """runAnalysis is the API that performs piecewise linear / piecewise constant analysis on a given teamplate."""
    
    # Initialize the data frames and templates to plot.
    dfs = []
    plotTemplates=[]
    
    # Run the analysis for each template upto the template count.
    for t in templates:
        try:
            # get the signal dataframe for specified room and signals
            dataDF = get_signal_dataframe(room, [t], hour_mean=False)

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
        except:
            print "Exception for template: ", t

    # Return the templates and dataframes.
    return [dfs, plotTemplates]


# In[ ]:



