# Databricks notebook source
# MAGIC %md The purpose of this notebook is to generate the features required for our segmentation work using a combination of feature engineering and dimension reduction techniques. This notebook has been developed on a Databricks ML 8.0 CPU-based cluster. 

# COMMAND ----------

# DBTITLE 1,Install Required Python Libraries
# MAGIC %pip install dython

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from sklearn.preprocessing import quantile_transform

import dython
import math

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md ## Step 1: Derive Bases Features
# MAGIC 
# MAGIC With a stated goal of segmenting customer households based on their responsiveness to various promotional efforts, we start by calculating the number of purchase dates (*pdates\_*) and the volume of sales (*amount\_list_*) associated with each promotion item, alone and in combination with one another.  The promotional items considered are:
# MAGIC 
# MAGIC * Campaign targeted products (*campaign\_targeted_*)
# MAGIC * Private label products (*private\_label_*)
# MAGIC * InStore-discounted products (*instore\_discount_*)
# MAGIC * Campaign (retailer-generated) coupon redemptions (*campaign\_coupon\_redemption_*)
# MAGIC * Manufacturer-generated coupon redemptions (*manuf\_coupon\_redemption_*)
# MAGIC 
# MAGIC The resulting metrics are by no means exhaustive but provide a useful starting point for our analysis:

# COMMAND ----------

# DBTITLE 1,Derive Relevant Metrics
# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS journey.household_metrics;
# MAGIC 
# MAGIC CREATE VIEW journey.household_metrics
# MAGIC AS
# MAGIC   WITH 
# MAGIC     targeted_products_by_household AS (
# MAGIC       SELECT DISTINCT
# MAGIC         b.household_id,
# MAGIC         c.product_id
# MAGIC       FROM journey.campaigns a
# MAGIC       INNER JOIN journey.campaigns_households b
# MAGIC         ON a.campaign_id=b.campaign_id
# MAGIC       INNER JOIN journey.coupons c
# MAGIC         ON a.campaign_id=c.campaign_id
# MAGIC       ),
# MAGIC     product_spend AS (
# MAGIC       SELECT
# MAGIC         a.household_id,
# MAGIC         a.product_id,
# MAGIC         a.day,
# MAGIC         a.basket_id,
# MAGIC         CASE WHEN a.campaign_coupon_discount > 0 THEN 1 ELSE 0 END as campaign_coupon_redemption,
# MAGIC         CASE WHEN a.manuf_coupon_discount > 0 THEN 1 ELSE 0 END as manuf_coupon_redemption,
# MAGIC         CASE WHEN a.instore_discount > 0 THEN 1 ELSE 0 END as instore_discount_applied,
# MAGIC         CASE WHEN b.brand = 'Private' THEN 1 ELSE 0 END as private_label,
# MAGIC         a.amount_list,
# MAGIC         a.campaign_coupon_discount,
# MAGIC         a.manuf_coupon_discount,
# MAGIC         a.total_coupon_discount,
# MAGIC         a.instore_discount,
# MAGIC         a.amount_paid  
# MAGIC       FROM journey.transactions_adj a
# MAGIC       INNER JOIN journey.products b
# MAGIC         ON a.product_id=b.product_id
# MAGIC       )
# MAGIC   SELECT
# MAGIC 
# MAGIC     x.household_id,
# MAGIC 
# MAGIC     -- Purchase Date Level Metrics
# MAGIC     COUNT(DISTINCT x.day) as purchase_dates,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL THEN x.day ELSE NULL END) as pdates_campaign_targeted,
# MAGIC     COUNT(DISTINCT CASE WHEN x.private_label = 1 THEN x.day ELSE NULL END) as pdates_private_label,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.private_label = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_private_label,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemptions,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.private_label = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC     COUNT(DISTINCT CASE WHEN x.manuf_coupon_redemption = 1 THEN x.day ELSE NULL END) as pdates_manuf_coupon_redemptions,
# MAGIC     COUNT(DISTINCT CASE WHEN x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.manuf_coupon_redemption = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC 
# MAGIC     -- List Amount Metrics
# MAGIC     COALESCE(SUM(x.amount_list),0) as amount_list,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.amount_list),0) as amount_list_with_campaign_targeted,
# MAGIC     COALESCE(SUM(x.private_label * x.amount_list),0) as amount_list_with_private_label,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.private_label * x.amount_list),0) as amount_list_with_campaign_targeted_private_label,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.amount_list),0) as amount_list_with_campaign_coupon_redemptions,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.private_label * x.amount_list),0) as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC     COALESCE(SUM(x.manuf_coupon_redemption * x.amount_list),0) as amount_list_with_manuf_coupon_redemptions,
# MAGIC     COALESCE(SUM(x.instore_discount_applied * x.amount_list),0) as amount_list_with_instore_discount_applied,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.manuf_coupon_redemption * x.instore_discount_applied * x.amount_list),0) as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC 
# MAGIC   FROM product_spend x
# MAGIC   LEFT OUTER JOIN targeted_products_by_household y
# MAGIC     ON x.household_id=y.household_id AND x.product_id=y.product_id
# MAGIC   GROUP BY 
# MAGIC     x.household_id;
# MAGIC     
# MAGIC SELECT * FROM journey.household_metrics;

# COMMAND ----------

# MAGIC %md It is assumed that the households included in this dataset were selected based on a minimum level of activity spanning the 711 day period over which data is provided.  That said, different households demonstrate different levels of purchase frequency during his period as well as different levels of overall spend.  In order to normalize these values between households, we'll divide each metric by the total purchase dates or total list amount associated with that household over its available purchase history:
# MAGIC 
# MAGIC **NOTE** Normalizing the data based on total purchase dates and spend as we do in this next step may not be appropriate in all analyses. 

# COMMAND ----------

# DBTITLE 1,Convert Metrics to Features
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS journey.household_features;
# MAGIC 
# MAGIC CREATE VIEW journey.household_features 
# MAGIC AS 
# MAGIC 
# MAGIC SELECT
# MAGIC       household_id,
# MAGIC   
# MAGIC       pdates_campaign_targeted/purchase_dates as pdates_campaign_targeted,
# MAGIC       pdates_private_label/purchase_dates as pdates_private_label,
# MAGIC       pdates_campaign_targeted_private_label/purchase_dates as pdates_campaign_targeted_private_label,
# MAGIC       pdates_campaign_coupon_redemptions/purchase_dates as pdates_campaign_coupon_redemptions,
# MAGIC       pdates_campaign_coupon_redemptions_on_private_labels/purchase_dates as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       pdates_manuf_coupon_redemptions/purchase_dates as pdates_manuf_coupon_redemptions,
# MAGIC       pdates_instore_discount_applied/purchase_dates as pdates_instore_discount_applied,
# MAGIC       pdates_campaign_targeted_instore_discount_applied/purchase_dates as pdates_campaign_targeted_instore_discount_applied,
# MAGIC       pdates_private_label_instore_discount_applied/purchase_dates as pdates_private_label_instore_discount_applied,
# MAGIC       pdates_campaign_targeted_private_label_instore_discount_applied/purchase_dates as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       pdates_campaign_coupon_redemption_instore_discount_applied/purchase_dates as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       pdates_campaign_coupon_redemption_private_label_instore_discount_applied/purchase_dates as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       pdates_manuf_coupon_redemption_instore_discount_applied/purchase_dates as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC       
# MAGIC       amount_list_with_campaign_targeted/amount_list as amount_list_with_campaign_targeted,
# MAGIC       amount_list_with_private_label/amount_list as amount_list_with_private_label,
# MAGIC       amount_list_with_campaign_targeted_private_label/amount_list as amount_list_with_campaign_targeted_private_label,
# MAGIC       amount_list_with_campaign_coupon_redemptions/amount_list as amount_list_with_campaign_coupon_redemptions,
# MAGIC       amount_list_with_campaign_coupon_redemptions_on_private_labels/amount_list as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC       amount_list_with_manuf_coupon_redemptions/amount_list as amount_list_with_manuf_coupon_redemptions,
# MAGIC       amount_list_with_instore_discount_applied/amount_list as amount_list_with_instore_discount_applied,
# MAGIC       amount_list_with_campaign_targeted_instore_discount_applied/amount_list as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC       amount_list_with_private_label_instore_discount_applied/amount_list as amount_list_with_private_label_instore_discount_applied,
# MAGIC       amount_list_with_campaign_targeted_private_label_instore_discount_applied/amount_list as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       amount_list_with_campaign_coupon_redemption_instore_discount_applied/amount_list as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied/amount_list as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       amount_list_with_manuf_coupon_redemption_instore_discount_applied/amount_list as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC 
# MAGIC FROM journey.household_metrics
# MAGIC ORDER BY household_id;
# MAGIC 
# MAGIC SELECT * FROM journey.household_features;

# COMMAND ----------

# MAGIC %md ## Step 2: Examine Distributions
# MAGIC 
# MAGIC Before proceeding, it's a good idea to examine our features closely to understand their compatibility with clustering techniques we might employ. In general, our preference would be to have standardized data with relatively normal distributions though that's not a hard requirement for every clustering algorithm.
# MAGIC 
# MAGIC To help us examine data distributions, we'll pull our data into a pandas Dataframe.  If our data volume were too large for pandas, we might consider taking a random sample (using the [*sample()*](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sample) against the Spark DataFrame) to examine the distributions:

# COMMAND ----------

# DBTITLE 1,Retrieve Features
# retreive as Spark dataframe
household_features = (
  spark
    .table('journey.household_features')
  )

# retrieve as pandas Dataframe
household_features_pd = household_features.toPandas()

# collect some basic info on our features
household_features_pd.info()

# COMMAND ----------

# MAGIC %md Notice that we have elected to retrieve the *household_id* field with this dataset.  Unique identifiers such as this will not be passed into the data transformation and clustering work that follows but may be useful in helping us validate the results of that work. By retrieving this information with our features, we can now separate our features and the unique identifier into two separate pandas dataframes where instances in each can easily be reassociated leveraging a shared index value:

# COMMAND ----------

# DBTITLE 1,Separate Household ID from Features
# get household ids from dataframe
households_pd = household_features_pd[['household_id']]

# remove household ids from dataframe
features_pd = household_features_pd.drop(['household_id'], axis=1)

features_pd

# COMMAND ----------

# MAGIC %md Let's now start examining the structure of our features:

# COMMAND ----------

# DBTITLE 1,Summary Stats on Features
features_pd.describe()

# COMMAND ----------

# MAGIC %md A quick review of the features finds that many have very low means and a large number of zero values (as indicated by the occurrence of zeros at multiple quantile positions).  We should take a closer look at the distribution of our features to make sure we don't have any data distribution problems that will trip us up later:

# COMMAND ----------

# DBTITLE 1,Examine Feature Distributions
feature_names = features_pd.columns
feature_count = len(feature_names)

# determine required rows and columns for visualizations
column_count = 5
row_count = math.ceil(feature_count / column_count)

# configure figure layout
fig, ax = plt.subplots(row_count, column_count, figsize =(column_count * 4.5, row_count * 3))

# render distribution of each feature
for k in range(0,feature_count):
  
  # determine row & col position
  col = k % column_count
  row = int(k / column_count)
  
  # set figure at row & col position
  ax[row][col].hist(features_pd[feature_names[k]], rwidth=0.95, bins=10) # histogram
  ax[row][col].set_xlim(0,1)   # set x scale 0 to 1
  ax[row][col].set_ylim(0,features_pd.shape[0]) # set y scale 0 to 2500 (household count)
  ax[row][col].text(x=0.1, y=features_pd.shape[0]-100, s=feature_names[k].replace('_','\n'), fontsize=9, va='top')      # feature name in chart

# COMMAND ----------

# MAGIC %md A quick visual inspection shows us that we have *zero-inflated distributions* associated with many of our features.  This is a common challenge when a feature attempts to measure the magnitude of an event that occurs with low frequency.  
# MAGIC 
# MAGIC There is a growing body of literature describing various techniques for dealing with zero-inflated distributions and even some zero-inflated models designed to work with them.  For our purposes, we will simply separate features with these distributions into two features, one of which will capture the occurrence of the event as a binary (categorical) feature and the other which will capture the magnitude of the event when it occurs:
# MAGIC 
# MAGIC **NOTE** We will label our binary features with a *has\_* prefix to make them more easily identifiable. We expect that if a household has zero purchase dates associated with an event, we'd expect that household also has no sales amount values for that event. With that in mind, we will create a single binary feature for an event and a secondary feature for each of the associated purchase date and amount list values.

# COMMAND ----------

# DBTITLE 1,Define Features to Address Zero-Inflated Distribution Problem
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS journey.household_features;
# MAGIC 
# MAGIC CREATE VIEW journey.household_features 
# MAGIC AS 
# MAGIC 
# MAGIC SELECT
# MAGIC 
# MAGIC       household_id,
# MAGIC       
# MAGIC       -- binary features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted,
# MAGIC       -- CASE WHEN pdates_private_label > 0 THEN 1 ELSE 0 END as has_pdates_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_manuf_coupon_redemptions,
# MAGIC       -- CASE WHEN pdates_instore_discount_applied > 0 THEN 1 ELSE 0 END as has_pdates_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_instore_discount_applied,
# MAGIC       -- CASE WHEN pdates_private_label_instore_discount_applied > 0 THEN 1 ELSE 0 END as has_pdates_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC   
# MAGIC       -- purchase date features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN pdates_campaign_targeted/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted,
# MAGIC       pdates_private_label/purchase_dates as pdates_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN pdates_campaign_targeted_private_label/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN pdates_campaign_coupon_redemptions/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN pdates_campaign_coupon_redemptions_on_private_labels/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN pdates_manuf_coupon_redemptions/purchase_dates 
# MAGIC         ELSE NULL END as pdates_manuf_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN pdates_campaign_targeted_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_instore_discount_applied,
# MAGIC       pdates_private_label_instore_discount_applied/purchase_dates as pdates_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN pdates_campaign_targeted_private_label_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN pdates_campaign_coupon_redemption_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN pdates_manuf_coupon_redemption_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC       
# MAGIC       -- list amount features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN amount_list_with_campaign_targeted/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted,
# MAGIC       amount_list_with_private_label/amount_list as amount_list_with_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN amount_list_with_campaign_targeted_private_label/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN amount_list_with_campaign_coupon_redemptions/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN amount_list_with_campaign_coupon_redemptions_on_private_labels/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN amount_list_with_manuf_coupon_redemptions/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_manuf_coupon_redemptions,
# MAGIC       amount_list_with_instore_discount_applied/amount_list as amount_list_with_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN amount_list_with_campaign_targeted_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC       amount_list_with_private_label_instore_discount_applied/amount_list as amount_list_with_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN amount_list_with_campaign_targeted_private_label_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN amount_list_with_campaign_coupon_redemption_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN amount_list_with_manuf_coupon_redemption_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC 
# MAGIC FROM journey.household_metrics
# MAGIC ORDER BY household_id;

# COMMAND ----------

# DBTITLE 1,Read Features to Pandas
# retreive as Spark dataframe
household_features = (
  spark
    .table('journey.household_features')
  )

# retrieve as pandas Dataframe
household_features_pd = household_features.toPandas()

# get household ids from dataframe
households_pd = household_features_pd[['household_id']]

# remove household ids from dataframe
features_pd = household_features_pd.drop(['household_id'], axis=1)

features_pd

# COMMAND ----------

# MAGIC %md With our features separated, let's look again at our feature distributions.  We'll start by examining our new binary features:

# COMMAND ----------

# DBTITLE 1,Examine Distribution of Binary Features
b_feature_names = list(filter(lambda f:f[0:4]==('has_') , features_pd.columns))
b_feature_count = len(b_feature_names)

# determine required rows and columns
b_column_count = 5
b_row_count = math.ceil(b_feature_count / b_column_count)

# configure figure layout
fig, ax = plt.subplots(b_row_count, b_column_count, figsize =(b_column_count * 3.5, b_row_count * 3.5))

# render distribution of each feature
for k in range(0,b_feature_count):
  
  # determine row & col position
  b_col = k % b_column_count
  b_row = int(k / b_column_count)
  
  # determine feature to be plotted
  f = b_feature_names[k]
  
  value_counts = features_pd[f].value_counts()

  # render pie chart
  ax[b_row][b_col].pie(
    x = value_counts.values,
    labels = value_counts.index,
    explode = None,
    autopct='%1.1f%%',
    labeldistance=None,
    #pctdistance=0.4,
    frame=True,
    radius=0.48,
    center=(0.5, 0.5)
    )
  
  # clear frame of ticks
  ax[b_row][b_col].set_xticks([])
  ax[b_row][b_col].set_yticks([])
  
  # legend & feature name
  ax[b_row][b_col].legend(bbox_to_anchor=(1.04,1.05),loc='upper left', fontsize=8)
  ax[b_row][b_col].text(1.04,0.8, s=b_feature_names[k].replace('_','\n'), fontsize=8, va='top')
