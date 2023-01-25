# Databricks notebook source
# MAGIC %pip install dbdemos

# COMMAND ----------

import dbdemos
dbdemos.help()
dbdemos.list_demos()

dbdemos.install('lakehouse-retail-c360', path='./', overwrite = True)

# COMMAND ----------

dbdemos.install('mlops-end2end')


# COMMAND ----------


