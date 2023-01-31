# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC This notebook shows you how to create and query a table or DataFrame loaded from data stored in Azure Blob storage.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 1: Set the data location and type
# MAGIC 
# MAGIC There are two ways to access Azure Blob storage: account keys and shared access signatures (SAS).
# MAGIC 
# MAGIC To get started, we need to set the location and type of the file.

# COMMAND ----------

storage_account_name = "grbox"
storage_account_access_key = "8RbQ8SRZAXH380f+raVuBO5UCZHScAHOKdOoFjs7MHE6x9axFbrwK9UsDYZ7P79KD3Set9gJnX5+mGOUZOSPcg=="
container = "gavi"

# COMMAND ----------

file_location = "wasbs://german@grbox.blob.core.windows.net"
file_type = "csv"

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# COMMAND ----------

dbutils.fs.ls(file_location)

# COMMAND ----------

# MAGIC %fs ls wasbs://wids2022@grbox.blob.core.windows.net/

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 2: Read the data
# MAGIC 
# MAGIC Now that we have specified our file metadata, we can create a DataFrame. Notice that we use an *option* to specify that we want to infer the schema from the file. We can also explicitly set this to a particular schema if we have one already.
# MAGIC 
# MAGIC First, let's create a DataFrame in Python.

# COMMAND ----------

df = (spark.read.format(file_type)
    .option("header", "true")
    .option("inferSchema", "true")
    .load("wasbs://datasets@grbox.blob.core.windows.net/housing/boston.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3: Create a table
# MAGIC 
# MAGIC If you want to query this data as a table, you can simply register it as a *view* or a table.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Since this table is registered as a temp view, it will be available only to this notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.

# COMMAND ----------

df.write.format("delta").saveAsTable("german_credit_data")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This table will persist across cluster restarts and allow various users across different notebooks to query this data.
