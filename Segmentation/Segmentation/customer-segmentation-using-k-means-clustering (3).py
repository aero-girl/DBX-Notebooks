# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 In this dataset, each entry represents a person who takes a credit by a bank.<br>
# MAGIC     📌 This project aims to break down credit card users of a bank into segments which can help the bank to understand their credit card users. <br>
# MAGIC     📌 We will segment our data with K-means clustering, an algorithm for clustering unlabeled dataset.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Importing Libraries 📚
# MAGIC <div style="font-family: cursive; line-height: 2; font-size:18px">
# MAGIC     📌 <b>Importing libraries</b> that will be used in this notebook.
# MAGIC </div>

# COMMAND ----------

# base
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Create List of Color Palletes 
color_mix = ['#03045e', '#023e8a', '#0077b6', '#0096c7','#00b4d8', '#48cae4', '#90e0ef','#A5E6F3', '#caf0f8']

# warning
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# STEP 1: RUN THIS CELL TO INSTALL BAMBOOLIB

# You can also install bamboolib on the cluster. Just talk to your cluster admin for that
%pip install bamboolib

# COMMAND ----------

# STEP 2: RUN THIS CELL TO IMPORT AND USE BAMBOOLIB

import bamboolib as bam

# This opens a UI from which you can import your data
bam  

# Already have a pandas data frame? Just display it!
# Here's an example
# import pandas as pd
# df_test = pd.DataFrame(dict(a=[1,2]))
# df_test  # <- You will see a green button above the data set if you display it

# COMMAND ----------

# STEP 1: RUN THIS CELL TO INSTALL BAMBOOLIB

# You can also install bamboolib on the cluster. Just talk to your cluster admin for that
%pip install bamboolib

# COMMAND ----------

# STEP 2: RUN THIS CELL TO IMPORT AND USE BAMBOOLIB

import bamboolib as bam

# This opens a UI from which you can import your data
bam  

# Already have a pandas data frame? Just display it!
# Here's an example
# import pandas as pd
# df_test = pd.DataFrame(dict(a=[1,2]))
# df_test  # <- You will see a green button above the data set if you display it

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Importing Dataset ⏬

# COMMAND ----------

data = spark.table("gavita.german_credit_data_1_csv").toPandas()
data.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC 📌 <b>A visual </b> of the first <b>five rows</b> in the dataset.</div>

# COMMAND ----------

data.head().style.background_gradient(cmap = 'Blues').set_properties(**{'font-family': 'Cursive'})

# COMMAND ----------

# *** Print Dataset Info ***
print('*' * 30)
print('** Dataset Info **')
print('*' * 30)
print('Total Rows:', data.shape[0])
print('Total Columns:', data.shape[1])

print('\n')

# *** Print Dataset Detail ***
print('*' * 30 )
print('** Dataset Details **')
print('*' * 30 )
data.info(memory_usage = False)

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC 📌 In the dataset, there are <b>10 columns</b> and <b>1000 observations</b> with various data types.<br>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC 📌 The first column is simply an index which we can delete.
# MAGIC </div>

# COMMAND ----------

data.drop(data.columns[0], inplace=True, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Data Exploration 🔍
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 This section will focused on <b>initial data exploration</b> before pre-process the data.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Categorical Variable 🔠</div>
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC   📌 This section will focused on identifying <b>categorical data</b>.
# MAGIC </div>

# COMMAND ----------

# Print Categorical Columns 
print('*' * 30)
print('** Categorical Columns **')
print('*' * 30)

categorical = []
for c in data.columns:
    if data[c].dtype == 'object':
        categorical += [c]        

for i in categorical:
    print(i)



# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Numerical Variables 🔠
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 This section will focused on identifying <b>numerical data</b>.
# MAGIC </div>

# COMMAND ----------

# Print Numerical Columns 
print('*' * 30)
print('** Numerical Columns **')
print('*' * 30)

numerical = []
for n in data.columns:
    if data[n].dtype == 'int64' or data[n].dtype == 'int32' :
        numerical += [n]
       
for i in numerical:
    print(i)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2.1 Descriptive Statistics 📏
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 Let's have a look at the <b>descriptive statistics</b> of numerical variables.
# MAGIC </div>

# COMMAND ----------

# Descriptive Statistics 
data[numerical].describe().T.style.background_gradient(cmap = 'Blues').set_properties(**{'font-family': 'Cursive'})

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2.2 Distribution of Numerical Variables 📊
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 This section will show the numerical column distribution using <b>histograms and box plots</b>.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. EDA 🔍📈👓
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 This section will perform some <b>EDA</b> to get more insights about dataset.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 Distribution of Numerical Variables 📊

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.1 Age

# COMMAND ----------

# Variable, Color & Plot Size 
var = data['Age']
color = color_mix[0]
fig = plt.figure(figsize = (14, 10))

# Skewness & Kurtosis 
print('*' * 40)
print('** Age Skewness & Kurtosis**')
print('*' * 40)
print('Skewness: {:.3f}'.format(var.skew(axis = 0, skipna = True)))
print('Kurtosis: {:.3f}'.format(var.kurt(axis = 0, skipna = True)))
print('\n')

# General Title
fig.suptitle('Age Distribution', fontweight = 'bold', fontsize = 16)
fig.subplots_adjust(top = 0.9)

#  Histogram 
ax_1=fig.add_subplot(1, 2, 2)
plt.title('Histogram Plot', fontweight = 'bold', fontsize = 14)
sns.histplot(data = data, x = var, kde = True, color = color)
plt.xlabel('Age', fontweight = 'regular', fontsize = 11)
plt.ylabel('Total', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

#  Box Plot 
ax_3 = fig.add_subplot(1, 2, 1)
plt.title('Box Plot', fontweight = 'bold', fontsize = 14)
sns.boxplot(y = var, data = data, color = color, linewidth = 1.5)
plt.ylabel('Age', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'y', alpha = 0.2)
plt.show();
