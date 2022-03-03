#!/usr/bin/env python
# coding: utf-8

# ### DATA ANALYSIS WITH PYTHON 

# In[1]:


# import pandas library

import pandas as pd
import numpy as np


# In[2]:


# Read the online file by the URL provides above, and assign it to variable "df"

other_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(other_path, header=None)


# In[3]:


print ("The first 5 rows of the dataframe")
df.head(5)


# #### Question #1: 
# ##### Check the bottom 10 rows of data frame "df".

# In[4]:


print("The last 10 rows of the dataframe") 
df.tail(10)


# #### Adding Headers Manually

# In[5]:


# create headers list

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "lenght", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
print("headers\n", headers)


# #### We replace headers and recheck our dataframe:

# In[6]:


df.columns = headers 
df.head(10)


# #### We need to replace the "?" symbol with NaN so the dropna() can remove the missing values

# In[7]:


df1=df.replace('?',np.NaN)


# #### We can drop missing values along the column "price" as follows:

# In[8]:


df=df1.dropna(subset=["price"], axis=0)
df.head(20)


# #### Question #2: 
# #### Find the name of the columns of the dataframe.

# In[9]:


print(df.columns)


# #### Saving Dataset In CSV 

# In[10]:


df.to_csv("automobile.csv", index=False)


# In[11]:


# Data Types 

df.dtypes


# In[12]:


print(df.dtypes)


# #### Discribing Your Data 

# In[13]:


df.describe()


# ### Question #3: 
# ### You can select the columns of a dataframe by indicating the name of each column. For example, you can select the three columns as follows:
# 
# ### dataframe[[' column 1 ',column 2', 'column 3']]
# 
# ### Where "column" is the name of the column, you can apply the method ".describe()" to get the statistics of those columns as follows:
# 
# ### dataframe[[' column 1 ',column 2', 'column 3'] ].describe()
# 
# ### Apply the method to ".describe()" to the columns 'length' and 'compression-ratio'.

# In[14]:


df[['lenght', 'compression-ratio']].describe()


# ### Info
# ### Another method you can use to check your dataset is:

# In[15]:


df.info()


# In[16]:


import pandas as pd 
import matplotlib.pylab as plt


# In[17]:


# Importing file from IBM Cloud 
filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"


# In[18]:


#Creating a Python list headers containing name of headers.

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


# In[19]:


df = pd.read_csv(filename, names =headers)


# ### Use the method head() to display the first five rows of the dataframe.

# In[20]:


df.head(5)


# ### Identify and handle missing values
# #### Identify missing values

# In[21]:


import numpy as np

# replace "?" to NaN
df.replace("?", np.nan, inplace= True)
df.head(5)


# ### Evaluating for Missing Data
# #### The missing values are converted by default. We use the following functions to identify these missing values. There are two methods to detect missing data:
# 
# #### .isnull()
# #### .notnull()

# In[22]:


missing_data = df.isnull()
missing_data.head(5)


# ### Count missing values in each column
# #### Using a for loop in Python, we can quickly figure out the number of missing values in each column. 
# #### As mentioned above, "True" represents a missing value and "False" means the value is present in the dataset. 
# #### In the body of the for loop the method ".value_counts()" counts the number of "True" values.

# In[23]:


for column in missing_data.columns.values.tolist(): 
    print(column)
    print(missing_data[column].value_counts())
    print("")


# #### Calculate the mean value for the "normalized-losses" column 

# In[24]:


avg_norm_loss = df["normalized-losses"].astype(float).mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)


# #### Replace "NaN" with mean value in "normalized-losses" column

# In[25]:


df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace = True)


# #### Calculate the mean value for the "bore" column 

# In[26]:


avg_bore =df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)


# In[27]:


df["bore"].replace(np.nan, avg_bore, inplace=True)


# #### Calculate the mean value for the "horsepower" column

# In[28]:


avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)


# #### Replace "NaN" with the mean value in the "horsepower" column

# In[29]:


df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)


# #### Calculate the mean value for the "stroke" column

# In[30]:


#Calculate the mean vaule for "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)

# replace NaN by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_stroke, inplace = True)


# #### Calculate the mean value for "peak-rpm" column

# In[31]:


avg_peakrpm =df['peak-rpm'].astype("float").mean(axis=0)
print ("Average of peak rpm", avg_peakrpm)


# In[32]:


df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace= True)


# #### To see which values are present in a particular column, we can use the ".value_counts()" method:

# In[33]:


df['num-of-doors'].value_counts()


# #### We can see that four doors are the most common type. We can also use the ".idxmax()" method to calculate the most common type automatically:

# In[34]:


df['num-of-doors'].value_counts().idxmax()


# #### The replacement procedure is very similar to what we have seen previously:
# ##### replace the missing 'num-of-doors' values by the most frequent 
#     

# In[35]:


df["num-of-doors"].replace(np.nan, "four", inplace = True)


# #### Finally, let's drop all rows that do not have price data:
# 

# In[36]:


# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace =True)


# In[37]:


df.head()


# In[38]:


df.dtypes


# #### Convert data types to proper format

# In[39]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]]= df[["normalized-losses"]].astype("int")
df[["price"]] =df[["price"]].astype("float")
df[["peak-rpm"]] =df[["peak-rpm"]].astype("float")


# In[40]:


df.dtypes


# ### Data Standardization

# In[41]:


df.head()


# #### Convert mpg to L/100km by mathematical operation (235 divided by mpg)

# In[42]:


df['city-L/100km'] = 235/df["city-mpg"]
df.head()


# #### According to the example above, transform mpg to L/100km in the column of "highway-mpg" and change the name of column to "highway-L/100km".

# In[43]:


df['highway-L/100km'] = 235/df["highway-mpg"]
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
df.head()


# ### Data Normalization
# #### Why normalization?
# 
# ##### Normalization is the process of transforming values of several variables into a similar range. Typical normalizations include scaling the variable so the variable average is 0, scaling the variable so the variance is 1, or scaling the variable so the variable values range from 0 to 1.

# #### replace (original value) by (original value)/(maximum value)

# In[44]:


df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()


# In[45]:


df['height'] = df['height']/df['height'].max()


# In[46]:


#show the scaled columns

df[["length", "width", "height"]].head()


# #### Convert data to correct format:

# In[47]:


df["horsepower"] =df["horsepower"].astype(int, copy=True)


# ### Let's plot the histogram of horsepower to see what the distribution of horsepower looks like.

# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# ##### We would like 3 bins of equal size bandwidth so we use numpy's linspace(start_value, end_value, numbers_generated function.

# In[49]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# #### We set group names:

# In[50]:


group_names =['Low', 'Medium', 'High']


# In[51]:


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest = True)
df[['horsepower','horsepower-binned']].head(20)


# ### Let's see the number of vehicles in each bin:

# In[52]:


df["horsepower-binned"].value_counts()


# ### Bins Visualization
# #### Normally, a histogram is used to visualize the distribution of bins we created above.

# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot


# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# #### Indicator Variable (or Dummy Variable)
# ##### What is an indicator variable?
# ###### An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning.
# 
# ##### We use indicator variables so we can use categorical variables for regression analysis in the later modules.

# In[54]:


df.columns


# #### Get the indicator variables and assign it to data frame "dummy_variable_1":

# In[55]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# #### Change the column names for clarity:

# In[56]:


dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel': 'fuel-type-diesel'}, inplace =True)
dummy_variable_1.head()


# #### In the dataframe, column 'fuel-type' has values for 'gas' and 'diesel' as 0s and 1s now.

# In[57]:


# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)


# In[58]:


df.head()


# In[59]:


# Write your code below and press Shift+Enter to execute 
# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()


# In[60]:


# merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)


# In[61]:


df.to_csv('clean_df.csv')


# In[ ]:





# In[ ]:




