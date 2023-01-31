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

# MAGIC %md
# MAGIC # 3. Importing Dataset ⏬

# COMMAND ----------

data_0 = spark.table("german_credit").toPandas()

# COMMAND ----------

data = data_0.drop(data_0.columns[0], inplace=False, axis=1)
data

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
# MAGIC ## 5.3 Analysis of Categorical Variables 🆎

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3.1 Sex

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
# MAGIC     📌 Most applicants own their home</b>.
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
