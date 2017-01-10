
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')


# In[3]:

import os
import sys

# For Mac OS X and Linux
#spark_path = "/opt/spark"
# For Windows 7+
spark_path = "C:/opt/spark"

os.environ['SPARK_HOME'] = spark_path
os.environ['HADOOP_HOME'] = spark_path

sys.path.append(spark_path + "/bin")
sys.path.append(spark_path + "/python")
sys.path.append(spark_path + "/python/pyspark/")
sys.path.append(spark_path + "/python/lib")
sys.path.append(spark_path + "/python/lib/pyspark.zip")
sys.path.append(spark_path + "/python/lib/py4j-0.10.3-src.zip")

from pyspark import SparkContext
from pyspark import SparkConf

sc = SparkContext()


# In[4]:

# import pyspark
# sc = pyspark.SparkContext()


# In[6]:

import pyspark
ss = pyspark.sql.SparkSession(sc)
spark = ss.builder      .master("local")      .appName("Word Count")      .config("spark.some.config.option", "some-value")      .getOrCreate()
sqlCtx = pyspark.SQLContext(sc)


# In[11]:

df = spark.read.csv('data/rm4226.csv', header=True)
df1 = spark.read.csv('data/rm2138.csv', header=True)
df2 = spark.read.csv('data/allbuildingsensordata.csv', header=True)

df.registerTempTable('rm4226')
df1.registerTempTable('rm2138')
df2.registerTempTable('allbuildingsensordata')


# In[12]:

from pyspark.sql.types import *
from pyspark.sql import Row
import ast
import json
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import os
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)
from pyspark.sql import HiveContext
hiveContext=HiveContext(sc)
import sys
from numpy import *


# In[22]:

#Model Classes
# %load PieceWise.py
#Encoder class for spark dataframes
import pandas as pd
from numpy import *
class encoder:
    """
    The encoder/decoder class is the base class for all encoder/decoder pairs.
    Subclasses encode different types of encoding.
    EncoderLearner is a factory class for fitting encoders to data
    """
    def __init__(self,raw,max_gap):
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
        """
        Get size of compressed representation
        """
        return len(self.compressed)
    
    def compute_error(self,S,compressed):
        """
        Compute mean sq error for compressed representation
        """        
        if type(compressed)==type(None):
            compressed=self.compressed
        #R=self.recon(compressed=compressed,index=S.index)
        R=self.recon()
        V=R-S
        V=V.dropna()
        return sqrt(sum([v*v for v in V.values]))/len(V)
      
class piecewise_constant(encoder):
    """ 
    Represent the signal using a sequence of piecewise constant functions 
    """
    def __init__(self,raw,max_gap):
      S=raw
      if type(S) != pd.Series:
        raise 'encode expects pandas Series as input'
      self.index=S.index
      self.Sol=self.fit(S,max_gap)    
    def fit(self,S,max_gap):
        """
        fit uses dynamic programming to find the best piecewise constant solution
        """	
        S[np.isnan(S)]=0
        _range=np.max(S)-np.min(S)
        #_range is a constant that is added to the error at each stop point
        #Larger values will cause fewer switches.
        print 'range=',_range
        #Dynamic programming
        Sol=[[]]*len(S)  # an array that holds the best partition ending at each point of the sequence.
                # Each element contains a best current value, a pointer to the last change in best 
                # solution so far and the total error of best solution so far.
        for i in range(len(S)):
            if i==0:
                Sol[i]={'prev':None, 'value':S[0], 'error':0.0, 'switch_no':0}
            # Sol is indexed by the location in the sequence S
            # prev: the index of the last switch point
            # value: current prediction value
            # error: cumulative error to this point
            # switch_no: number of switches so far.
            else:
                err0 = Sol[i-1]['error']+(Sol[i-1]['value']-S[i])**2
                best=None
                best_err=1e20
                best_val=S[i]
                for j in xrange(np.max([0,i-max_gap]),i):
                      _mean=np.mean(S[j:i])
                      _std=np.std(S[j:i])
                      err=_std*(i-j)+Sol[j]['error']+_range
                      if err<best_err:
                          best=j
                          best_val=_mean
                          best_err=err
                Sol[i]={'prev':best, 'value':best_val, 'error':best_err,                        'switch_no': Sol[best]['switch_no']+1}
            #print '\r',i,Sol[i],
        return Sol
    
    def compress(self,S):
        """
        compress the data
        """
        Switch_points=[]
        i=len(self.Sol)-1                # start from the end 
        while i>0:
            prev=self.Sol[i]['prev']
            value=self.Sol[i]['value']
            if self.Sol[prev]['value'] != value:
                Switch_points.append({'time':S.index[prev],'value':value})
            i=prev
        self.compressed=Switch_points
        return Switch_points

    def recon(self,compressed=None, index=None):
        #print '\nindex=',index==None,'\n'
        #print '\ncompressed=',compressed==None,'\n'
        """
        reconstruct compressed representation
        """
        if(type(index)==type(None)):
            index=self.index
        Recon=pd.Series(index=index)
        
        if(type(compressed)==type(None)):
            compressed=self.compressed
        for e in compressed:
            time=e['time']
            value=e['value']
            Recon[time]=value
            
        return Recon.fillna(method='ffill')
      
class piecewise_linear(encoder):
    """ 
    Represent the signal using a sequence of piecewise linear functions 
    """
    def __init__(self,raw,max_gap):
      S=raw
      if type(raw) != pd.Series:
        raise 'encode expects pandas Series as input'
      self.index=raw.index
      self.Sol=self.fit(raw,max_gap)
    
    # fit uses dynamic programming to find the best piecewise linear solution
    # max_gap is the maximal extent of a single step.
    # Reason for max_gap is that even if the error is small we want to correct
    # it with some minimal frequence. 
    # Not quite a snapshot because the value will not necessarily change after 
    # max_gap is reached.
    def fit(self,S,max_gap):
        """
        fit uses dynamic programming to find the best piecewise linear solution
        """
        S[np.isnan(S)]=0
        _range=np.max(S)-np.min(S)
        # _range is a constant that is added to the error at each stop point
        # Larger values will cause fewer switches.
        print 'range=',_range
        #Dynamic programming
        Sol=[[]]*len(S)  # an array that holds the best partition ending at each point of the sequence.
                # Each element contains a best current value, a pointer to the last change in best 
                # solution so far and the total error of best solution so far.
        for i in range(len(S)):
            if i==0:
                Sol[i]={'prev':None, 'value':S[0], 'error':0.0, 'switch_no':0, 'slope':0}
            # Sol is indexed by the location in the sequence S
            # prev: the index of the last switch point
            # value: current prediction value
            # error: cumulative error to this point
            # switch_no: number of switches so far.
            #slope: slope of th linear line at this point
            else:
                err0 = Sol[i-1]['error']+(Sol[i-1]['value']-S[i])**2
                best=None
                best_err=1e20
                best_val=S[i]
                best_slope=1e20
                for j in xrange(np.max([0,i-max_gap]),i):
                    #_mean=np.mean(S[j:i])
                    _slope=(S[i]-S[j])*1.0/(i-j)
                    #_std=np.std(S[j:i]
                    _val=0
                    _err=0
                    for k in xrange(j,i):
                      _val=Sol[j]['value']+_slope*(k-j)
                      _err+=(Sol[k]['value']-_val)**2
                    err=_err*1.0/(i-j)+Sol[j]['error']+_range
                    _val=Sol[j]['value']+_slope*(i-j)
                    if err<best_err:
                        best=j
                        best_val=_val
                        best_err=err
                        best_slope=_slope
                Sol[i]={'prev':best, 'value':best_val, 'error':best_err,                        'switch_no': Sol[best]['switch_no']+1, 'slope':best_slope}
            #print '\r',i,Sol[i],
        return Sol
    
    def compress(self,S):
        """
        compress the piecewise linear data
        """
        Switch_points=[]
        i=len(self.Sol)-1                # start from the end 
        while i>0:
            prev=self.Sol[i]['prev']
            slope=self.Sol[i]['slope']
            value=self.Sol[i]['value']
            if self.Sol[prev]['slope'] != slope:
                Switch_points.append({'time':S.index[prev],'value':value})
            i=prev
        self.compressed=Switch_points
        return Switch_points

    def recon(self,compressed=None, index=None):
        #print '\nindex=',index==None,'\n'
        #print '\ncompressed=',compressed==None,'\n'
        """
        reconstruct the compressed piecewise linear data
        """
        print "HI"
        if(type(index)==type(None)):
            index=self.index
        Recon=pd.Series(index=index)
        if(type(compressed)==type(None)):
            compressed=self.compressed
        for e in compressed:
            time=e['time']
            value=e['value']
            Recon[time]=value
        
        #print Recon    
        #return Recon.fillna(method='ffill')
        Recon.interpolate(method="linear", inplace=True)
        return Recon


# In[38]:

#All applicable functions
def runAnalysisOld(room,template,stTime,enTime,method,tolerance):
    """
    run old analysis on room, template, start and end time, method and tolerance
    """
    dataDF=loadf(room,template,stTime,enTime)
    print len(dataDF)
    [compressedDF, reconDF]=model(dataDF,method,tolerance)
    return [dataDF, compressedDF, reconDF]
  
def loadf(room,template,stTime,enTime):
    """
    load data with room, template, start and end time
    """    
    query = "select * from allbuildingsensordata where room='"+str(room)+"' and template='"+str(template)+"' and timeseries between '"+ str(stTime) +"' and '"+str(enTime)+"'"
      #print query
      #df=hiveContext.sql(query)
    df=spark.sql(query)
    dataDF=pd.DataFrame(df.select('timeseries','values').collect(),columns=['timeseries','values'])
    dataDF['timeseries']=dataDF['timeseries'].apply(lambda x:datetime.strptime(x, '%Y-%m-%dT%H:%M:%S+00:00'))
    dataDF['values']=dataDF['values'].apply(lambda x:float(str(x)))
    return dataDF
    
def dataPlot(dfArgument,timeLowerBound, timUpperBound):
    """
    data plot for the arguement with lower and upper time boundary
    """
    tempDF=dfArgument[dfArgument.timeseries.between(timeLowerBound, timUpperBound)]
  #if(Rooms!='All'):
  #  tempDF=tempDF[tempDF['room']==Rooms]
  #display(tempDF)
    dataDF=pd.DataFrame(tempDF.select('timeseries','values').limit(5000).collect(),columns=['timeseries','values'])
    return dataDF

def model(A,method,tolerance):
    """
    use the model based on the method and tolerance
    """
    pd_df=A
    sys.stdout.flush()
  #pd_df=pd.DataFrame(A.select('timeseries','values').limit(5000).collect(),columns=['timeseries','values'])
  #pd_df.plot(kind='line')
    S=pd_df['values']
    _std=np.std(S)
    print "Std dev is ",_std
    if(method=='piecewise_constant'):
        encoder=piecewise_constant(S,tolerance)
    elif(method=='piecewise_linear'):
        encoder=piecewise_linear(S,tolerance)
    C=encoder.compress(S)
    R=encoder.recon()
    print "type is ",type(R)
    compressed_df=pd.DataFrame(C)
    print 'size=',encoder.get_size(),
    error=encoder.compute_error(S,compressed=C)
    print 'error=',error, 'error/_std=',error/_std
    print C
    return [compressed_df, R]
  
def runAnalysis(table,stTime,enTime,templateCount):
      """
      run analysis based on the template on the data with start and end time
      """
      templates=['Zone Temperature','Actual Supply Flow','Occupied Clg Min','Occupied Htg Flow','Common Setpoint', 'Actual Heating Setpoint', 'Supply Vel Press', 'Zone Temperature Error',  'Damper Position',  'Warm Cool Adjust', 'Cooling Command', 'HVAC Zone Power', 'Damper Command', 'Cooling Max Flow', 'Occupied Htg Flow','Actual Cooling Setpoint', 'Reheat Valve Command Error']
      dfs=[]
      plotTemplates=[]
      for t in templates[0:templateCount]:
        try:
          query = "select * from "+str(table)+" where template='"+str(t)+"' and timeseries between '"+ str(stTime) +"' and '"+str(enTime)+"'" 
          print query
          #df=hiveContext.sql(query)
          df = spark.sql(query)
          #df = global_df
          dataDF=pd.DataFrame(df.select('timeseries','values').collect(),columns=['timeseries','values'])
          dataDF['timeseries']=dataDF['timeseries'].apply(lambda x:datetime.strptime(x, '%Y-%m-%dT%H:%M:%S+00:00'))
          dataDF['values']=dataDF['values'].apply(lambda x:float(str(x)))
          if(t in ['Zone Temperature','Zone Temperature Error']):
            method='piecewise_linear'
          else:
            method='piecewise_constant'
          [compressedDF, reconDF]=model(dataDF,method,tolerance=96)
          plotTemplates.append(t)
          dfs.extend([dataDF,reconDF]) 
        except:
          print "rajeshb", t
      return [dfs,plotTemplates]
    
def getTime(x,dfTest):
    """
    get the timeseries from the data
    """
    return dfTest.at[int(x),'timeseries']

def plotResults(dfs,plotTemplates):
      """
      plot templates on the compression analysis
      """
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
          print 2*i
          ReconDF=dfs[2*i+1].to_frame(name='values')
          ReconDF['time']=ReconDF.index
          ReconDF=ReconDF.dropna()
          ReconDF=ReconDF.sort(['time'], ascending=[1])
          ReconDF['timeseries']=ReconDF.apply(lambda x: getTime(x['time'],dfs[2*i]), axis=1)
          if(plotTemplates[i] in ['Actual Supply Flow','Occupied Command','Damper Position']):
            axes1Count=1
            axes[1].plot(dfs[2*i]['timeseries'].tolist(),dfs[2*i]['values'].tolist(),'k--',color=colors[i],label=plotTemplates[i])
            axes[1].plot(ReconDF['timeseries'].tolist(),ReconDF['values'].tolist(),'k:',color=colors[i],label=plotTemplates[i]+'_reconstructed')
          else:
            axes0Count=1
            axes[0].plot(dfs[2*i]['timeseries'].tolist(),dfs[2*i]['values'].tolist(),'k--',color=colors[i],label=plotTemplates[i])
            axes[0].plot(ReconDF['timeseries'].tolist(),ReconDF['values'].tolist(),'k:',color=colors[i],label=plotTemplates[i]+'_reconstructed')
        except:
          print plotTemplates[i]
      if(axes1Count==0):
        print "axes1Count",axes1Count
        print len(timeX)
        axes[1].plot(timeX,[0 for x in timeX],'k',color='w',label='axes1')
      if(axes0Count==0):
        print "axes0Count",axes0Count
        axes[0].plot(timeX,[0 for x in timeX],'k',color='w',label='axes0')
      axes[0].legend(loc='upper left',fontsize='x-small')
      axes[1].legend(loc='upper right',fontsize='x-small')
      axes[1].set_ylabel('scale for top right legend')
      axes[0].set_ylabel('scale for top left legend')
      #display(fig)
  
def loadTags():
    """
    load tags.json
    """
    tagsDF=sqlContext.read.parquet("/mnt/assignment1/csvParquetFiles/tagsDF")
    keep=[tagsDF.name,tagsDF.sensor_id,tagsDF.template,tagsDF.timeseries_span]
    tagsDF=tagsDF.select(*keep)
    return tagsDF

def help():	
	print "class encoder : " +encoder.__doc__
	print "	encoder.compress :" + encoder.compress.__doc__
	print "	encoder.recon :" + encoder.recon.__doc__
	print "	encoder.get_size :" + encoder.get_size.__doc__	
	print "	encoder.compute_error :" + encoder.compute_error.__doc__		

	print "class piecewise_constant : " +piecewise_constant.__doc__	
	print "	piecewise_constant.fit :" + piecewise_constant.fit.__doc__
	print "	piecewise_constant.compress :" + piecewise_constant.compress.__doc__
	print "	piecewise_constant.recon :"+ piecewise_constant.recon.__doc__
	
	print "class piecewise_linear : " +piecewise_linear.__doc__	
	print "	piecewise_linear.fit :" +piecewise_linear.fit.__doc__
	print "	piecewise_linear.compress :" +piecewise_linear.compress.__doc__
	print "	piecewise_linear.recon :" +piecewise_linear.recon.__doc__

	print "runAnalysisOld :"+runAnalysisOld.__doc__
	print "loadf :"+loadf.__doc__
	print "dataPlot :"+dataPlot.__doc__
	print "model :"+model.__doc__
	print "runAnalysis :"+runAnalysis.__doc__
	print "getTime :"+getTime.__doc__
	print "plotResults :"+plotResults.__doc__

	print "loadTags : "  +loadTags.__doc__

