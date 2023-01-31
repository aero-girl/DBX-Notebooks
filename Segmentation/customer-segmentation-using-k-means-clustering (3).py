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

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; text-align: justify; font-size:18px">
# MAGIC     📌 From the plots, it can be seen that Age is <b>skewed</b>. <br>
# MAGIC     📌 The kurtosis value is 0.60, which indicates that Age has a lower tail, and distribution stretched around the center.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC   <blockquote >📝 <b>Negative values</b> for the skewness indicate data that are skewed left (left tail is long) and <b>positive values</b> for the skewness indicate data that are skewed right (right tail is long). If skewness is <b>between -1 and -0.5 or between 0.5 and 1</b>, the distribution is <b>moderately skewed</b>. If skewness is <b>between -0.5 and 0.5</b>, the distribution is <b>approximately symmetric</b>.
# MAGIC </blockquote>
# MAGIC     <blockquote >📝 Kurtosis is a measure of whether the data are heavy-tailed (outliers) or light-tailed (lack of outliers) relative to a normal distribution. <b>Kurtosis</b> values used to show <b>tailedness of a column</b>. The value of normal distribution <b>mesokurtotic</b> should be equal to 3. If kurtosis value is more than 3, it is called <b>leptokurtic</b>, meaning the distribution has very long and skinny tails. Meanwhile, if kurtosis value is less than 3, then it is called <b>platikurtic</b>, meaning the distribution has a lower tail and stretched around center tails.</blockquote>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.2 Job

# COMMAND ----------

# Variable, Color & Plot Size 
var = data['Job']
color = color_mix[1]
fig = plt.figure(figsize = (14, 10))

# Skewness & Kurtosis 
print('*' * 40)
print('** Job Skewness & Kurtosis **')
print('*' * 40)
print('Skewness: {:.3f}'.format(var.skew(axis = 0, skipna = True)))
print('Kurtosis: {:.3f}'.format(var.kurt(axis = 0, skipna = True)))
print('\n')

# General Title 
fig.suptitle('Job Distribution', fontweight = 'bold', fontsize = 16)
fig.subplots_adjust(top = 0.9)

# Histogram 
ax_1=fig.add_subplot(1, 2, 2)
plt.title('Histogram Plot', fontweight = 'bold', fontsize = 14)
sns.histplot(data = data, x = var, kde = True, color = color)
plt.xlabel('Job', fontweight = 'regular', fontsize = 11)
plt.ylabel('Total', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

#  Box Plot 
ax_2 = fig.add_subplot(1, 2, 1)
plt.title('Box Plot', fontweight = 'bold', fontsize = 14)
sns.boxplot(y = var, data = data, color = color, linewidth = 1.5)
plt.ylabel('Job', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'y', alpha = 0.2)
plt.show();



# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; text-align: justify; font-size:18px">
# MAGIC     📌 The numeric legend for Job is as follows: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled). <br>
# MAGIC     📌 From the plots, it can be seen that most applicants are <b>skilled</b>.    
# MAGIC </div>

# COMMAND ----------

data

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.3 Credit amount

# COMMAND ----------

#  Variable, Color & Plot Size 
var = data['Credit_amount']
color = color_mix[2]
fig = plt.figure(figsize = (14, 10))

# Skewness & Kurtosis 
print('*' * 40)
print('** Credit Skewness & Kurtosis**')
print('*' * 40)
print('Skewness: {:.3f}'.format(var.skew(axis = 0, skipna = True)))
print('Kurtosis: {:.3f}'.format(var.kurt(axis = 0, skipna = True)))
print('\n')

#  General Title
fig.suptitle('Credit Amount Distribution', fontweight = 'bold', fontsize = 16)
fig.subplots_adjust(top = 0.9)

#  Histogram 
ax_1=fig.add_subplot(1, 2, 2)
plt.title('Histogram Plot', fontweight = 'bold', fontsize = 14)
sns.histplot(data = data, x = var, kde = True, color = color)
plt.xlabel('Credit Amount', fontweight = 'regular', fontsize = 11)
plt.ylabel('Total', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

#  Box Plot 
ax_2 = fig.add_subplot(1, 2, 1)
plt.title('Box Plot', fontweight = 'bold', fontsize = 14)
sns.boxplot(y = var, data = data, color = color, linewidth = 1.5)
plt.ylabel('Credit Amount', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'y', alpha = 0.2)
plt.show();

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; text-align: justify; font-size:18px">
# MAGIC     📌 Credit amount is <b>positively skewed</b>. <br>
# MAGIC     📌 The kurtosis value is more than 3 suggest that Credit amount is leptokurtic, meaning the distribution has very long and skinny tails. In other words, it is more susceptible to <b>outliers</b>.  
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.4 Duration

# COMMAND ----------

# Variable, Color & Plot Size 
var = data['Duration']
color = color_mix[3]
fig = plt.figure(figsize = (14, 10))

# Skewness & Kurtosis 
print('*' * 40)
print('** Duration Skewness & Kurtosis**')
print('*' * 40)
print('Skewness: {:.3f}'.format(var.skew(axis = 0, skipna = True)))
print('Kurtosis: {:.3f}'.format(var.kurt(axis = 0, skipna = True)))
print('\n')

#  General Title 
fig.suptitle('Duration Distribution', fontweight = 'bold', fontsize = 16)
fig.subplots_adjust(top = 0.9)

#  Histogram 
ax_1=fig.add_subplot(1, 2, 2)
plt.title('Histogram Plot', fontweight = 'bold', fontsize = 14)
sns.histplot(data = data, x = var, kde = True, color = color)
plt.xlabel('Duration', fontweight = 'regular', fontsize = 11)
plt.ylabel('Total', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

# Box Plot 
ax_2 = fig.add_subplot(1, 2, 1)
plt.title('Box Plot', fontweight = 'bold', fontsize = 14)
sns.boxplot(y = var, data = data, color = color, linewidth = 1.5)
plt.ylabel('Duration', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'y', alpha = 0.2)
plt.show();

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; text-align: justify; font-size:18px">
# MAGIC     📌 Duration is also <b>positively skewed</b>, albeit moderately. <br>
# MAGIC     📌 Duration is distributed from 4 to 72 months  
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 Heatmap 🔥 and Pair Plot 📈

# COMMAND ----------

# Compute correlation
corr = data.corr(method = 'pearson')

# Convert correlation to numpy array
mask = np.array(corr)

# Mask the repetitive value for each pair
mask[np.tril_indices_from(mask)] = False

# Correlation Map (Heatmap) 
plt.figure(figsize = (12, 8))
sns.heatmap(corr, mask=mask, square = True, annot = True, cmap = 'Blues', linewidths = 0.1)
plt.suptitle('Correlation Map', fontweight = 'heavy', fontsize = 14)
plt.tight_layout(rect = [0, 0.04, 1, 1.01])

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; text-align: justify; font-size:18px">
# MAGIC     📌 Most features in the dataset are strongly correlated to each other. <br>
# MAGIC     📌 Duration and Credit amount are highly correlated.
# MAGIC </div>

# COMMAND ----------

# Plot the pairwise relationships in a dataset
plt.figure(figsize = (12, 8))
sns.pairplot(data)
plt.suptitle('Pair Plot', fontweight = 'heavy', fontsize = 14)
plt.tight_layout(rect = [0, 0.04, 1, 1.01])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 Analysis of Categorical Variables 🆎

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3.1 Sex

# COMMAND ----------

# Setting Colors, Labels, Order
colors=color_mix[0:9]
labels=data['Sex'].dropna().unique()
order=data['Sex'].value_counts().index

# Size for Both Figures
plt.figure(figsize=(18, 8))
plt.suptitle('Sex', fontweight='heavy', fontsize='16')

# Histogram 
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14)
ax = sns.countplot(x='Sex', data=data, palette=colors, order=order, alpha=0.85)
for rect in ax.patches:
     ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+10,rect.get_height(), horizontalalignment='center', 
             fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.xlabel('Sex', fontweight='bold', fontsize=11)
plt.ylabel('Total', fontweight='bold', fontsize=11)
plt.grid(axis='y', alpha=0.4)
countplt

# Pie Chart 
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14)
plt.pie(data['Sex'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor='black'), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white')
plt.gcf().gca().add_artist(centre);
    
    
# Print "Sex" Values 
print('*' * 30)
print('Sex')
print('*' * 30)
data.Sex.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; text-align: justify; font-size:18px">
# MAGIC     📌 There are twice as much males compared to females <br>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3.2 Housing

# COMMAND ----------

# Setting Labels, Order
labels=data['Housing'].dropna().unique()
order=data['Housing'].value_counts().index

# Size for Both Figures
plt.figure(figsize=(18, 8))
plt.suptitle('Housing', fontweight='heavy', fontsize='16')

# Histogram 
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14)
ax = sns.countplot(x='Housing', data=data, palette=colors, order=order, alpha=0.85)
for rect in ax.patches:
     ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+10,rect.get_height(), horizontalalignment='center', 
             fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.xlabel('Housing', fontweight='bold', fontsize=11)
plt.ylabel('Total', fontweight='bold', fontsize=11)
plt.grid(axis='y', alpha=0.4)
countplt

# Pie Chart 
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14)
plt.pie(data['Housing'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor='black'), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white')
plt.gcf().gca().add_artist(centre);
    
    
# Print "Sex" Values 
print('*' * 30)
print('Housing')
print('*' * 30)
data.Housing.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 Most applicants town their home</b>.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3.3 Saving accounts

# COMMAND ----------

# Setting Labels, Order
labels=data['Saving_accounts'].dropna().unique()
order=data['Saving_accounts'].value_counts().index

# Size for Both Figures
plt.figure(figsize=(18, 8))
plt.suptitle('Saving_accounts', fontweight='heavy', fontsize='16')

# Histogram 
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14)
ax = sns.countplot(x='Saving_accounts', data=data, palette=colors, order=order, alpha=0.85)
for rect in ax.patches:
     ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+10,rect.get_height(), horizontalalignment='center', 
             fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.xlabel('Saving_accounts', fontweight='bold', fontsize=11)
plt.ylabel('Total', fontweight='bold', fontsize=11)
plt.grid(axis='y', alpha=0.4)
countplt

# Pie Chart 
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14)
plt.pie(data['Saving_accounts'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor='black'), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white')
plt.gcf().gca().add_artist(centre);
    
    
# Print "Saving accounts" Values 
print('*' * 30)
print('Saving_accounts')
print('*' * 30)
data['Saving_accounts'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 Most applicants have little savings</b>.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3.4 Checking account

# COMMAND ----------

# Setting Labels, Order
labels=data['Checking_account'].dropna().unique()
order=data['Checking_account'].value_counts().index

# Size for Both Figures
plt.figure(figsize=(18, 8))
plt.suptitle('Checking_account', fontweight='heavy', fontsize='16')

# Histogram 
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14)
ax = sns.countplot(x='Checking_account', data=data, palette=colors, order=order, alpha=0.85)
for rect in ax.patches:
     ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+5,rect.get_height(), horizontalalignment='center', 
             fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.xlabel('Checking_account', fontweight='bold', fontsize=11)
plt.ylabel('Total', fontweight='bold', fontsize=11)
plt.grid(axis='y', alpha=0.4)
countplt

# Pie Chart 
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14)
plt.pie(data['Checking_account'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor='black'), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white')
plt.gcf().gca().add_artist(centre);
    
    
# Print "Checking account" Values 
print('*' * 30)
print('Checking_account')
print('*' * 30)
data['Checking_account'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 Most applicants have little in their checking account</b>.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3.5 Purpose

# COMMAND ----------

# Setting Labels, Order
labels=data['Purpose'].dropna().unique()
order=data['Purpose'].value_counts().index

# Size for Both Figures
plt.figure(figsize=(18, 8))
plt.suptitle('Purpose', fontweight='heavy', fontsize='16')

# Histogram 
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14)
ax = sns.countplot(y='Purpose', data=data, palette=colors, order=order, alpha=0.85)
for rect in ax.patches:
    width, height = rect.get_width(), rect.get_height()
    x, y = rect.get_xy()
    ax.text (x+width+10, y+height/2, '{:.0f}'.format(width), horizontalalignment='center', verticalalignment='center')
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.xlabel('Purpose', fontweight='bold', fontsize=11)
plt.ylabel('Total', fontweight='bold', fontsize=11)
plt.grid(axis='x', alpha=0.5)
countplt


# Pie Chart 
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14)
plt.pie(data['Purpose'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor='black'), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white')
plt.gcf().gca().add_artist(centre);
    
    
# Print "Purpose" Values 
print('*' * 30)
print('Purpose')
print('*' * 30)
data['Purpose'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 Most applicants obtain credit for the purpose of buying a car, followed by radio/TV</b>.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Data Pre-processing and Feature Engineering ⚙
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px" >
# MAGIC     📌 This section will process the dataset</b>.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 Missing Values Analysis

# COMMAND ----------

#  Total Missing Values in each Columns 
print('*' * 45)
print('** Total Missing Values in each Columns **')
print('*' * 45)
data.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 Most features in this dataset have no missing values. <br>
# MAGIC     📌 Saving accounts and Checking account have missing values. <br>
# MAGIC     📌 Saving accounts have 183 missing values.<br>
# MAGIC     📌 Checking account have 394 missing values.<br>
# MAGIC     📌 Missing values for Saving and Checking accounts are probably due to the applicants not having either one of the accounts.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 We will use the <b>fillna</b> imputation method for Saving and Checking accounts
# MAGIC </div>

# COMMAND ----------

data['Saving_accounts'] = data["Saving_accounts"].fillna("none")
data["Checking_account"] = data["Checking_account"].fillna("none")

# COMMAND ----------

#  Total Missing Values in each Columns 
print('*' * 45)
print('** Total Missing Values in each Columns after imputation**')
print('*' * 45)
data.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 Transform Data 🔗

# COMMAND ----------

# Define Color & Plot Size 
color = color_mix[0:7]
fig = plt.figure(figsize = (14, 10))

# Size for all figures
plt.suptitle('Visualisation of log transformation', fontweight='heavy', fontsize='16')

# Histogram 
ax_1 = fig.add_subplot(2, 3, 1)
plt.title('Histogram Plot', fontweight = 'bold', fontsize = 14)
sns.histplot(data = data, x = data['Age'], kde = True, color = color[0])
plt.xlabel('Age', fontweight = 'regular', fontsize = 11)
plt.ylabel('Total', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

ax_2 = fig.add_subplot(2, 3, 2)
plt.title('Histogram Plot', fontweight = 'bold', fontsize = 14)
sns.histplot(data = data, x = data['Credit_amount'], kde = True, color = color[2])
plt.xlabel('Credit_Amount', fontweight = 'regular', fontsize = 11)
plt.ylabel('Total', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

ax_3 = fig.add_subplot(2, 3, 3)
plt.title('Histogram Plot', fontweight = 'bold', fontsize = 14)
sns.histplot(data = data, x = data['Duration'], kde = True, color = color[4])
plt.xlabel('Duration', fontweight = 'regular', fontsize = 11)
plt.ylabel('Total', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

# Histogram after log transform

ax_1 = fig.add_subplot(2, 3, 4)
plt.title('After Log Transform', fontweight = 'bold', fontsize = 12)
sns.histplot(data = data, x = np.log(data['Age']), kde = True, color = color[0])
plt.xlabel('Age', fontweight = 'regular', fontsize = 11)
plt.ylabel('Total', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

ax_2 = fig.add_subplot(2, 3, 5)
plt.title('After Log Transform', fontweight = 'bold', fontsize = 12)
sns.histplot(data = data, x = np.log(data['Credit_amount']), kde = True, color = color[2])
plt.xlabel('Credit_Amount', fontweight = 'regular', fontsize = 11)
plt.ylabel('Total', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

ax_3 = fig.add_subplot(2, 3, 6)
plt.title('After Log Transform', fontweight = 'bold', fontsize = 12)
sns.histplot(data = data, x = np.log(data['Duration']), kde = True, color = color[4])
plt.xlabel('Duration', fontweight = 'regular', fontsize = 11)
plt.ylabel('Total', fontweight = 'regular', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2;">
# MAGIC     📌 We can see that the skewness of the distribution has been resolved by performing the transformation.<br>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 Feature Engineering 🛠

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📝 In this Feature Engineering section, I will be carrying out the followinf steps:
# MAGIC 
# MAGIC * Scale the features using log transform 
# MAGIC * Label encoding the categorical features
# MAGIC * Scaling the features using the standard scaler 
# MAGIC * Creating a subset dataframe for dimensionality reduction </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2.1 Log Transform 📊
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC    📌 Scale the features using log transform </div>

# COMMAND ----------

# Transform the data to correct the skewness
data['Age'] = np.log(data['Age'])
data['Credit_amount'] = np.log(data['Credit_amount'])
data['Duration'] = np.log(data['Duration'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2.2 Label encoding 🏷
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 We will use Label encoding to convert the categorical features into numerical features.<br>
# MAGIC </div>

# COMMAND ----------

encoder = LabelEncoder()
for label in categorical:
    data[label] = encoder.fit_transform(data[label])
    
#  Check that all categorical features are transformed into numeric features 
print('*' * 55)
print('** Categorical Features ---> Numerical Features **')
print('*' * 55)
data

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2.3 Standard Scaler ⚖
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 Scaling the features using the standard scaler.
# MAGIC </div>

# COMMAND ----------

# Scaling
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(cluster_scaled, columns=data.columns)
data_scaled.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2.4 PCA 📉
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 Perform dimensionality reduction by using PCA. <br>
# MAGIC     📌 Principal component analysis (PCA) is a technique for reducing the dimensionality of such datasets, increasing interpretability but at the same time minimising information loss. <br>
# MAGIC </div>

# COMMAND ----------

# Initiating PCA to reduce dimentions or features to 2
pca = PCA(n_components = 2)
pca.fit(data_scaled)
data_PCA = pd.DataFrame(pca.transform(data_scaled), columns=(["dim1","dim2"]))
data_PCA.describe().T

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. K-means Clustering </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC   📌 Our first attempt at clustering with make use of the K-means algorithm. K-means is a simple, popular algorithm for dividing instances into clusters around a pre-defined number of centroids (cluster centers). The algorithm works by generating an initial set of points within the space to serve as cluster centers. Instances are then associated with the nearest of these points to form a cluster, and the true center of the resulting cluster is re-calculated. The new centroids are then used to re-enlist cluster members, and the process is repeated until a stable solution is generated (or until the maximum number of iterations is exhausted). A quick demonstration run of the algorithm may produce a result as follows:
# MAGIC 
# MAGIC 
# MAGIC     📌 The elbow method is used to determine the optimal number of clusters in K-means clustering. The elbow method plots the value of the cost function produced by different values of clusters, K. <br>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.1 Demonstrate Cluster Assignment

# COMMAND ----------

import matplotlib.cm as cm

# initial cluster count
initial_n = 4
 
# train the model
initial_model = KMeans(
  n_clusters=initial_n,
  max_iter=1000
  )
 
# fit and predict dataset cluster assignment
init_clusters = initial_model.fit_predict(data_PCA)
 
# combine data with cluster assignments
labeled_X_pd = (
  pd.concat( 
    [data_PCA, pd.DataFrame(init_clusters,columns=['cluster'])],
    axis=1
    )
  )

# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=labeled_X_pd,
  x='dim1',
  y='dim2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / initial_n) for i in range(initial_n)],
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.2 Iterate over Potential Values of K
# MAGIC 
# MAGIC Our initial model run demonstrates the mechanics of generating a K-means clustering solution, but it also demonstrates some of the shortcomings of the approach. First, we need to specify the number of clusters. Setting the value incorrectly can force the creation of numerous smaller clusters or just a few larger clusters, neither of which may reflect what we may observe to be the more immediate and natural structure inherent to the data.
# MAGIC 
# MAGIC Second, the results of the algorithm are highly dependent on the centroids with which it is initialized. The use of the K-means++ initialization algorithm addresses some of these problems by better ensuring that initial centroids are dispersed throughout the populated space, but there is still an element of randomness at play in these selections that can have big consequences for our results.
# MAGIC 
# MAGIC To begin working through these challenges, we will generate a large number of model runs over a range of potential cluster counts. For each run, we will calculate the sum of squared distances between members and assigned cluster centroids (inertia) as well as a secondary metric (silhouette score) which provides a combined measure of inter-cluster cohesion and intra-cluster separation (ranging between -1 and 1 with higher values being better). Because of the large number of iterations we will perform, we will distribute this work across our Databricks cluster so that it can be concluded in a timely manner:
# MAGIC 
# MAGIC NOTE We are using a Spark RDD as a crude means of exhaustively searching our parameter space in a distributed manner. This is an simple technique frequently used for efficient searches over a defined range of values.

# COMMAND ----------

from sklearn.metrics import silhouette_score, silhouette_samples

#Iterate over Potential Values of K
# broadcast features so that workers can access efficiently
X_broadcast = sc.broadcast(data_PCA)
 
# function to train model and return metrics
def evaluate_model(n):
  model = KMeans( n_clusters=n, init='k-means++', n_init=1, max_iter=10000)
  clusters = model.fit(X_broadcast.value).labels_
  return n, float(model.inertia_), float(silhouette_score(X_broadcast.value, clusters))
 
 
# define number of iterations for each value of k being considered
iterations = (
  spark
    .range(100) # iterations per value of k
    .crossJoin( spark.range(2,21).withColumnRenamed('id','n')) # cluster counts
    .repartition(sc.defaultParallelism)
    .select('n')
    .rdd
    )
 
# train and evaluate model for each iteration
results_pd = (
  spark
    .createDataFrame(
      iterations.map(lambda n: evaluate_model(n[0])), # iterate over each value of n
      schema=['n', 'inertia', 'silhouette']
      ).toPandas()
    )
 
# remove broadcast set from workers
X_broadcast.unpersist()
 
display(results_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.3 Inertia over Cluster Count
# MAGIC Plotting inertia relative to n, i.e. the target number of clusters, we can see that the total sum of squared distances between cluster members and cluster centers decreases as we increase the number of clusters in our solution. Our goal is not to drive inertia to zero (which would be achieved if we made each member the center of its own, 1-member cluster) but instead to identify the point in the curve where the incremental drop in inertia is diminished. In our plot, we might identify this point as occurring somewhere between 3 and 4 clusters.

# COMMAND ----------

display(results_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.4 Silhouette Score over Cluster Count
# MAGIC While providing a second perspective, the plot of silhouette scores reinforces the notion that selecting a number of clusters for K-means is a bit subjective. Domain knowledge coupled with inputs from these and similar charts (such as a chart of the Gap statistic) may help point you towards an optimal cluster count but there are no widely-accepted, objective means of determining this value to date.
# MAGIC 
# MAGIC NOTE We need to be careful to avoid chasing the highest value for the silhouette score in the knee chart. Higher scores can be obtained with higher values of n by simply pushing outliers into trivially small clusters.
# MAGIC 
# MAGIC For our model, we'll go with a value of 2. Looking at the plot of inertia, there appears to be evidence supporting this value. Examining the silhouette scores, the clustering solution appears to be much more stable at this value than at values further down the range. To obtain domain knowledge, we might speak with our promotions experts and gain their perspective on not only how different households respond to promotions but what might be a workable number of clusters from this exercise. But most importantly, from our visualization, the presence of 2 well-separated clusters seems to naturally jump out at us.
# MAGIC 
# MAGIC With a value for n identified, we now need to generate a final cluster design. Given the randomness of the results we obtain from a K-means run (as captured in the widely variable silhouette scores), we might take a best-of-k approach to defining our cluster model. In such an approach, we run through some number of K-means model runs and select the run that delivers the best result as measured by a metric such as silhouette score. To distribute this work, we'll implement a custom function that will allow us to task each worker with finding a best-of-k solution and then take the overall best solution from the results of that work:
# MAGIC 
# MAGIC NOTE We are again using an RDD to allow us to distribute the work across our cluster. The iterations RDD will hold a value for each iteration to perform. Using mapPartitions() we will determine how many iterations are assigned to a given partition and then force that worker to perform an appropriately configured best-of-k evaluation. Each partition will send back the best model it could discover and then we will take the best from these.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.5 Identify Best of K Model

# COMMAND ----------

total_iterations = 1000
n_for_bestofk = 3
X_broadcast = sc.broadcast(data_PCA)

def find_bestofk_for_partition(partition):
   
  # count iterations in this partition
  n_init = sum(1 for i in partition)
  
  # perform iterations to get best of k
  model = KMeans( n_clusters=n_for_bestofk, n_init=n_init, init='k-means++', max_iter=10000).fit(X_broadcast.value)
  model.fit(X_broadcast.value)
  
  # score model
  score = float(silhouette_score(X_broadcast.value, model.labels_))
  
  # return (score, model)
  yield (score, model)


# build RDD for distributed iteration
iterations = sc.range(
              total_iterations, 
              numSlices= sc.defaultParallelism * 4
              ) # distribute work into fairly even number of partitions that allow us to track progress
                        
# retreive best of distributed iterations
bestofk_results = (
  iterations
    .mapPartitions(find_bestofk_for_partition)
    .sortByKey(ascending=False)
    .take(1)
    )[0]

# get score and model
bestofk_score = bestofk_results[0]
bestofk_model = bestofk_results[1]
bestofk_clusters = bestofk_model.labels_

# print best score obtained
print('Silhouette Score: {0:.6f}'.format(bestofk_score))

# combine dataset with cluster assignments
bestofk_labeled_X_pd = (
  pd.concat( 
    [data_PCA, pd.DataFrame(bestofk_clusters,columns=['cluster'])],
    axis=1
    )
  )
                        
# clean up 
X_broadcast.unpersist()

# COMMAND ----------

bestofk_model.cluster_centers_

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 We can now visualize our results to get a sense of how the clusters align with the structure of our data. <br>
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.6b Visualise Best of K Clusters

# COMMAND ----------

# visualise cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=bestofk_labeled_X_pd,
  x='dim1',
  y='dim2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / n_for_bestofk) for i in range(n_for_bestofk)],  # align colors with those used in silhouette plots
  legend='brief',
  alpha=0.5,
  ax = ax
  )
ax = sns.scatterplot(bestofk_model.cluster_centers_[:, 0], bestofk_model.cluster_centers_[:, 1], ax=ax) # add cluster centers
plt.show()
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

clusters_range = range(3,15)
inertia =[]

for c in clusters_range:
    kmeans = KMeans(n_clusters=c, init='k-means++', random_state=0, n_init=30, max_iter=100)
    clusters = kmeans.fit_predict(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(18, 8))
ax = sns.lineplot(clusters_range, inertia, palette=colors, marker='o', markersize=10, markeredgewidth=1, markeredgecolor='black')
plt.title('Elbow method', fontweight = 'bold', fontsize = 14)
plt.xlabel('Number of Clusters', fontweight = 'bold', fontsize = 11)
plt.ylabel('Inertia', fontweight = 'bold', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)



# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 In order to decide and validate best number of cluster, we can use Silhoutte method.<br>
# MAGIC     📌 Silhouette coefficients (as these values are referred to as) near +1 indicate that the sample is far away from the neighboring clusters.<br>
# MAGIC     📌 A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters.<br>
# MAGIC     📌 Negative values indicate that those samples might have been assigned to the wrong cluster.
# MAGIC </div>

# COMMAND ----------

from sklearn.metrics import silhouette_score, silhouette_samples

for n_clusters in clusters_range:
    km = KMeans (n_clusters=n_clusters)
    preds = km.fit_predict(data_scaled)
    centers = km.cluster_centers_

    score = silhouette_score(cluster_scaled, preds, metric='euclidean')
    print('*' * 50)
    print ("For n_clusters = {}, silhouette score is {:.2f}".format(n_clusters, score))
    print('*' * 50)

# COMMAND ----------

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0, n_init=30, max_iter=100)

# Fitting data in model and predicting
clusters = kmeans.fit_predict(data_scaled)
data_scaled['cluster'] = clusters


# COMMAND ----------

data_scaled

# COMMAND ----------

colors = None
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,6))
sns.scatterplot(x="Credit amount",y="Duration", hue="cluster", palette=colors, data=data_scaled, ax=ax1)
sns.scatterplot(x="Age",y="Credit amount", hue="cluster", palette=colors, data=data_scaled, ax=ax2)
sns.scatterplot(x="Age",y="Duration", hue="cluster", palette=colors, data=data_scaled, ax=ax3)
plt.tight_layout()

# COMMAND ----------

grouped_km = cluster_data.groupby(['cluster']).mean().round(1)
grouped_km

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 Cluster 0: Higher credit amount, middle-aged, long duration customers.<br>
# MAGIC     📌 Cluster 1: Lower credit amount, older, short duration customers.<br>
# MAGIC     📌 Cluster 2: Lower credit amount, younger, short duration customers.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px " >
# MAGIC     🌟 Thank you for reading my notebook. Please upvote if this was helpful in any way 👍🏽. If you have any comments/feedback/suggestions, please feel free to comment.🌟
# MAGIC </div>

# COMMAND ----------

import matplotlib.cm as cm



# COMMAND ----------


