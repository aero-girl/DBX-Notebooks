# Databricks notebook source
# MAGIC %md
# MAGIC ## 🔥 What is FLAML 🔥
# MAGIC 
# MAGIC     
# MAGIC <b>FLAML</b>, is short for Fast and Lightweight Automated Machine Learning library. It is an open-source Python library created by Microsoft researchers in 2021 for automated machine learning (AutoML). It is designed to be fast, efficient, and user-friendly, making it ideal for a wide range of applications. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test

# COMMAND ----------

# MAGIC %md
# MAGIC ##  🚶🏽‍♀️ A walkthrough: Here's an example of how easy FLAML could be used to build a churn prediction model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 🎯 The objective of this notebook is:
# MAGIC * Demonstrate how quickly you can develop a machine learning model using <b>FLAML</b> to be used as a a baseline model
# MAGIC * Predict customer churn by assessing their propensity or risk to churn, appliying <b>FLAML</b> to solve a classification problem

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Libraries
# MAGIC 
# MAGIC Import the necessary libraries 

# COMMAND ----------

# Install bamboolib 
%pip install bamboolib  

# FLAML
%pip install flaml

# COMMAND ----------

# Data manipulation and analysis
import pandas as pd

# Machine learning library for Python
from sklearn.model_selection import train_test_split

# Functions for creating visualisations
import matplotlib.pyplot as plt

# Library for working with large, multi-dimensional arrays and matrices of numerical data
import numpy as np

# Mlflow
import mlflow
import mlflow.sklearn

# FLAML
# %pip install flaml
import flaml
from flaml import AutoML
from flaml.ml import sklearn_metric_loss_score
from flaml.data import get_output_from_log

# Install bamboolib 
# %pip install bamboolib  
import bamboolib as bam

# COMMAND ----------

# Read data into pandas DataFrame
df = spark.table("wa_fn_usec__telco_customer_churn_csv")

# Transform to pandas
data_telco = df.toPandas()

# Print the first 5 rows of the DataFrame
display(data_telco.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔭 EDA with bamboolib
# MAGIC 
# MAGIC This is a new feature within Databricks that I wanted to try out! It's quick and provides a nice UI to visualise your data.

# COMMAND ----------

# This opens a UI from which you can import your data
bam  

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔎 Basic EDA

# COMMAND ----------

# Get a summary of a DataFrame object
data_telco.info()

# COMMAND ----------

# Get statistical information on numerical features
data_telco.describe()

# COMMAND ----------

# Looping through the columns to get unique values per column.
for i in data_telco.columns:
    print(f"Unique {i}'s count: {data_telco[i].nunique()}")
    print(f"{data_telco[i].unique()}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Meaning of Features
# MAGIC <br> By inspecting the columns and their unique values, a general understanding about the features can be build. The features can also be clustered into different categories:
# MAGIC 
# MAGIC **Classification labels**
# MAGIC <br>Churn - Whether the customer churned or not (Yes or No)
# MAGIC 
# MAGIC **Customer services booked**
# MAGIC * PhoneService - Whether the customer has a phone service or not (Yes, No)
# MAGIC * MultipleLines - Whether the customer has multiple lines or not (Yes, No, No phone service)
# MAGIC * InternetService - Customer’s internet service provider (DSL, Fiber optic, No)
# MAGIC * OnlineSecurity - Whether the customer has online security or not (Yes, No, No internet service)
# MAGIC * OnlineBackup - Whether the customer has online backup or not (Yes, No, No internet service)
# MAGIC * DeviceProtection - Whether the customer has device protection or not (Yes, No, No internet service)
# MAGIC * TechSupport - Whether the customer has tech support or not (Yes, No, No internet service)
# MAGIC * StreamingTV - Whether the customer has streaming TV or not (Yes, No, No internet service)
# MAGIC * StreamingMovies - Whether the customer has streaming movies or not (Yes, No, No internet service)
# MAGIC 
# MAGIC **Customer account information**
# MAGIC * Tenure - Number of months the customer has stayed with the company
# MAGIC * Contract - The contract term of the customer (Month-to-month, One year, Two year)
# MAGIC * PaperlessBilling - Whether the customer has paperless billing or not (Yes, No)
# MAGIC * PaymentMethod - The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# MAGIC * MonthlyCharges - The amount charged to the customer monthly
# MAGIC * TotalCharges - The total amount charged to the customer
# MAGIC 
# MAGIC **Customers demographic info**
# MAGIC * customerID - Customer ID
# MAGIC * Gender - Whether the customer is a male or a female
# MAGIC * SeniorCitizen - Whether the customer is a senior citizen or not (1, 0)
# MAGIC * Partner - Whether the customer has a partner or not (Yes, No)
# MAGIC * Dependents - Whether the customer has dependents or not (Yes, No)
# MAGIC     

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙ Feature Preprocessing

# COMMAND ----------

# No internet service and No phone service can be grouped as No, so replacing these with 'No'
data_telco.replace('No internet service','No',inplace = True)
data_telco.replace('No phone service','No',inplace=True)

# COMMAND ----------

# Replace Yes with 1 and No with 0
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    data_telco[col].replace({'Yes':1, 'No':0},inplace=True)

# COMMAND ----------

# Replace Female with 1 and Male with 0
data_telco['gender'].replace({'Female':1,'Male':0},inplace=True)

# COMMAND ----------

# Let's convert all remaining categorical variables into dummy variables
data_telco = pd.get_dummies(data = data_telco,columns=['InternetService','Contract','PaymentMethod'])
data_telco.head()

# COMMAND ----------

# Converting Total Charges to a numerical data type
data_telco["TotalCharges"] = pd.to_numeric(data_telco["TotalCharges"], errors='coerce')
data_telco.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC When 'TotalCharges' was converted to a numerical variable, 11 missing observations were revealed. However, upon further observation, it can be seen that these are <b>new customers</b> indicated by the tenure column, where the entries for tenure are 0's.

# COMMAND ----------

data_telco[data_telco["TotalCharges"].isnull()]

# COMMAND ----------

# For these new customers, it makes sense to replace the TotalCharges with 0
data_telco["TotalCharges"].fillna(value= 0, inplace = True)

# COMMAND ----------

# Generate new feature "Number_AdditionalServices" by summing up the number of add-on services consumed.
data_telco['Number_AdditionalServices'] = (data_telco[['OnlineSecurity', 'DeviceProtection', 'StreamingMovies', 'TechSupport', 'StreamingTV', 'OnlineBackup']] == 'Yes').sum(axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⭐ Dataset Preparation
# MAGIC 
# MAGIC The 'train_test_split' function is used to split the 'X' and 'y' variables into training and testing sets, with the test set comprising 20% of the data and the remaining 80% being used for training.

# COMMAND ----------

y = data_telco['Churn']
X = data_telco.drop(["customerID", "Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print('X_train.shape',X_train.shape)
print('y_train.shape',y_train.shape)
print('X_test.shape',X_test.shape)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔥 FLAML automl

# COMMAND ----------

automl = AutoML()

# Then, we train the model on the training data and evaluate it on the testing data.
automl_settings = {
    "time_budget": 60,  # total running time in seconds
    "metric": "accuracy",  # metric to optimize
    "task": "classification",  # task type
    "log_file_name": "churn_pipeline.log",  # flaml log file
    }

# COMMAND ----------

# Enable autolog()
mlflow.sklearn.autolog()

# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run():
    
    # The main flaml automl API
    automl.fit(X_train, y_train, **automl_settings)
   
    # Log model
    mlflow.sklearn.log_model(automl, 'automl')
    
    # Log model accuracy 
    mlflow.log_metric('accuracy', (1-automl.best_loss))

    # Log automl_settings
    mlflow.log_params(automl_settings)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 👩🏽‍💻 Retrieve and analyze the outcomes of AutoML.fit() 

# COMMAND ----------

# Retrieve best config
print('Best estimator:', automl.model.estimator)
print('Best hyperparmeter config:', automl.best_config)
print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
print('Best accuracy on validation data: {0:.4g}'.format(1-automl.best_loss))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💥 Compute Predictions of Testing Dataset 

# COMMAND ----------

# Predict on testing dataset
y_pred = automl.predict(X_test)

# COMMAND ----------

# Compute metric values on test dataset
print('accuracy', '=', 1 - sklearn_metric_loss_score('accuracy', y_pred, y_test))
print('roc_auc', '=', sklearn_metric_loss_score('roc_auc', y_pred, y_test))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Log history 
# MAGIC You can also save and plot the history:

# COMMAND ----------

time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = \
    get_output_from_log(filename=automl_settings['log_file_name'], time_budget=120)

for config in config_history:
    print(config)

# COMMAND ----------

# Plot figure creating a learning curve plot that shows how the model's performance on the validation set (measured by the validation r2 score) changes over time
plt.title('Learning Curve')
plt.xlabel('Wall Clock Time (s)')
plt.ylabel('Validation r2')
plt.scatter(time_history, 1 - np.array(valid_loss_history))
plt.step(time_history, 1 - np.array(best_valid_loss_history), where='post')
plt.grid()

# COMMAND ----------

# MAGIC %md
# MAGIC # 🤝🏽 Summary 
# MAGIC 
# MAGIC Using the FLAML library, we have learnt to build a model to predict churn with relatively high accuracy. Similarly, we can apply the same process and build a regression model, time-series forecasting and NLP tasks using this library.
# MAGIC     
# MAGIC  <b>Reference: </b>[FLAML: Getting Started]( https://microsoft.github.io/FLAML/docs/getting-started)
