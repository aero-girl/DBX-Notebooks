# Databricks notebook source
# MAGIC %md The purpose of this notebook is to access and prepare the data required for our segmentation work. This notebook has been developed on a Databricks ML 8.0 CPU-based cluster. 

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
from pyspark.sql.functions import min, max

# COMMAND ----------

# MAGIC %md ## Step 1: Access the Data
# MAGIC 
# MAGIC The purpose of this exercise is to demonstrate how a Promotions Management team interested in segmenting customer households based on promotion responsiveness might perform the analytics portion of their work.  The dataset we will use has been made available by Dunnhumby via Kaggle and is referred to as [*The Complete Journey*](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey). It consists of numerous files identifying household purchasing activity in combination with various promotional campaigns for about 2,500 households over a nearly 2 year period. The schema of the overall dataset may be represented as follows:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/segmentation_journey_schema3.png' width=500>
# MAGIC 
# MAGIC To make this data available for our analysis, it is downloaded, extracted and loaded to the *bronze* folder of a [cloud-storage mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) named */mnt/completejourney*.  From there, we might prepare the data as follows:

# COMMAND ----------

# DBTITLE 1,Create Database
# MAGIC %sql
# MAGIC 
# MAGIC DROP DATABASE IF EXISTS journey CASCADE;
# MAGIC CREATE DATABASE journey;

# COMMAND ----------

# DBTITLE 1,Transactions
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.transactions')

# expected structure of the file
transactions_schema = StructType([
  StructField('household_id', IntegerType()),
  StructField('basket_id', LongType()),
  StructField('day', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('quantity', IntegerType()),
  StructField('sales_amount', FloatType()),
  StructField('store_id', IntegerType()),
  StructField('discount_amount', FloatType()),
  StructField('transaction_time', IntegerType()),
  StructField('week_no', IntegerType()),
  StructField('coupon_discount', FloatType()),
  StructField('coupon_discount_match', FloatType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/mnt/completejourney/bronze/transaction_data.csv',
      header=True,
      schema=transactions_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/mnt/completejourney/silver/transactions')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.transactions 
    USING DELTA 
    LOCATION '/mnt/completejourney/silver/transactions'
    ''')

# show data
display(
  spark.table('journey.transactions')
  )

# COMMAND ----------

# DBTITLE 1,Products
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.products')

# expected structure of the file
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('manufacturer', StringType()),
  StructField('department', StringType()),
  StructField('brand', StringType()),
  StructField('commodity', StringType()),
  StructField('subcommodity', StringType()),
  StructField('size', StringType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/mnt/completejourney/bronze/product.csv',
      header=True,
      schema=products_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/mnt/completejourney/silver/products')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.products
    USING DELTA 
    LOCATION '/mnt/completejourney/silver/products'
    ''')

# show data
display(
  spark.table('journey.products')
  )

# COMMAND ----------

# DBTITLE 1,Households
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.households')

# expected structure of the file
households_schema = StructType([
  StructField('age_bracket', StringType()),
  StructField('marital_status', StringType()),
  StructField('income_bracket', StringType()),
  StructField('homeownership', StringType()),
  StructField('composition', StringType()),
  StructField('size_category', StringType()),
  StructField('child_category', StringType()),
  StructField('household_id', IntegerType())
  ])

# read data to dataframe
households = (
  spark
    .read
    .csv(
      '/mnt/completejourney/bronze/hh_demographic.csv',
      header=True,
      schema=households_schema
      )
  )

# make queriable for later work
households.createOrReplaceTempView('households')

# income bracket sort order
income_bracket_lookup = (
  spark.createDataFrame(
    [(0,'Under 15K'),
     (15,'15-24K'),
     (25,'25-34K'),
     (35,'35-49K'),
     (50,'50-74K'),
     (75,'75-99K'),
     (100,'100-124K'),
     (125,'125-149K'),
     (150,'150-174K'),
     (175,'175-199K'),
     (200,'200-249K'),
     (250,'250K+') ],
    schema=StructType([
            StructField('income_bracket_numeric',IntegerType()),
            StructField('income_bracket', StringType())
            ])
    )
  )

# make queriable for later work
income_bracket_lookup.createOrReplaceTempView('income_bracket_lookup')

# household composition sort order
composition_lookup = (
  spark.createDataFrame(
    [ (0,'Single Female'),
      (1,'Single Male'),
      (2,'1 Adult Kids'),
      (3,'2 Adults Kids'),
      (4,'2 Adults No Kids'),
      (5,'Unknown') ],
    schema=StructType([
            StructField('sort_order',IntegerType()),
            StructField('composition', StringType())
            ])
    )
  )

# make queriable for later work
composition_lookup.createOrReplaceTempView('composition_lookup')

# persist data with sort order data and a priori segments
(
  spark
    .sql('''
      SELECT
        a.household_id,
        a.age_bracket,
        a.marital_status,
        a.income_bracket,
        COALESCE(b.income_bracket_numeric, -1) as income_bracket_alt,
        a.homeownership,
        a.composition,
        COALESCE(c.sort_order, -1) as composition_sort_order,
        a.size_category,
        a.child_category
      FROM households a
      LEFT OUTER JOIN income_bracket_lookup b
        ON a.income_bracket=b.income_bracket
      LEFT OUTER JOIN composition_lookup c
        ON a.composition=c.composition
      ''')
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/mnt/completejourney/silver/households')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.households 
    USING DELTA 
    LOCATION '/mnt/completejourney/silver/households'
    ''')

# show data
display(
  spark.table('journey.households')
  )

# COMMAND ----------

# DBTITLE 1,Coupons
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.coupons')

# expected structure of the file
coupons_schema = StructType([
  StructField('coupon_upc', StringType()),
  StructField('product_id', IntegerType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/mnt/completejourney/bronze/coupon.csv',
      header=True,
      schema=coupons_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/mnt/completejourney/silver/coupons')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.coupons
    USING DELTA 
    LOCATION '/mnt/completejourney/silver/coupons'
    ''')

# show data
display(
  spark.table('journey.coupons')
  )

# COMMAND ----------

# DBTITLE 1,Campaigns
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.campaigns')

# expected structure of the file
campaigns_schema = StructType([
  StructField('description', StringType()),
  StructField('campaign_id', IntegerType()),
  StructField('start_day', IntegerType()),
  StructField('end_day', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/mnt/completejourney/bronze/campaign_desc.csv',
      header=True,
      schema=campaigns_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/mnt/completejourney/silver/campaigns')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.campaigns
    USING DELTA 
    LOCATION '/mnt/completejourney/silver/campaigns'
    ''')

# show data
display(
  spark.table('journey.campaigns')
  )

# COMMAND ----------

# DBTITLE 1,Coupon Redemptions
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.coupon_redemptions')

# expected structure of the file
coupon_redemptions_schema = StructType([
  StructField('household_id', IntegerType()),
  StructField('day', IntegerType()),
  StructField('coupon_upc', StringType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/mnt/completejourney/bronze/coupon_redempt.csv',
      header=True,
      schema=coupon_redemptions_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/mnt/completejourney/silver/coupon_redemptions')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.coupon_redemptions
    USING DELTA 
    LOCATION '/mnt/completejourney/silver/coupon_redemptions'
    ''')

# show data
display(
  spark.table('journey.coupon_redemptions')
  )
