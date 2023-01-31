# Databricks notebook source
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


data = spark.table("gavita.german_credit_data_1_csv").toPandas()
data.head()


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

data.drop(data.columns[0], inplace=True, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # <div style="font-family: Cursive; background-color: #03045e; color: #FFFFFF; padding: 12px; line-height: 1.5;">4. Data Exploration 🔍</div>
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 This section will focused on <b>initial data exploration</b> before pre-process the data.
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
# MAGIC ## <div style="font-family: Cursive; background-color: #023e8a; color: #FFFFFF; padding: 12px; line-height: 1.5;">4.2 Numerical Variables 🔠</div>
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
  if data[n].dtype == 'int32':
    numerical += [n]
        
for j in numerical:
    print(j)

# COMMAND ----------

# Descriptive Statistics 
data[numerical].describe().T.style.background_gradient(cmap = 'Blues').set_properties(**{'font-family': 'Cursive'})

# COMMAND ----------


