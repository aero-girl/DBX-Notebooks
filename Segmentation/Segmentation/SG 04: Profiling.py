# Databricks notebook source
# MAGIC %md The purpose of this notebook is to better understand the clusters generated in the prior notebook leveraging some standard profiling techniques. This notebook has been developed on a Databricks ML 8.0 CPU-based cluster.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import mlflow

import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic

import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from pyspark.sql.functions import expr

# COMMAND ----------

# MAGIC %md ## Step 1: Assemble Segmented Dataset
# MAGIC 
# MAGIC We now have clusters but we're not really clear on what exactly they represent.  The feature engineering work we performed to avoid problems with the data that might lead us to invalid or inappropriate solutions have made the data very hard to interpret.  
# MAGIC 
# MAGIC To address this problem, we'll retrieve the cluster labels (assigned to each household) along with the original features associated with each:

# COMMAND ----------

# DBTITLE 1,Retrieve Features & Labels
# retrieve features and labels
household_basefeatures = spark.table('journey.household_features')
household_finalfeatures = spark.table('DELTA.`/mnt/completejourney/silver/features_finalized/`')
labels = spark.table('DELTA.`/mnt/completejourney/gold/household_clusters/`')

# assemble labeled feature sets
labeled_basefeatures_pd = (
  labels
    .join(household_basefeatures, on='household_id')
  ).toPandas()

labeled_finalfeatures_pd = (
  labels
    .join(household_finalfeatures, on='household_id')
  ).toPandas()

# get name of all non-feature columns
label_columns = labels.columns

labeled_basefeatures_pd

# COMMAND ----------

# MAGIC %md Before proceeding with our analysis of these data, let's set a few variables that will be used to control the remainder of our analysis.  We have multiple cluster designs but for this notebook, we will focus our attention on the results from our hierarchical clustering model:

# COMMAND ----------

# DBTITLE 1,Set Cluster Design to Analyze
cluster_column = 'hc_cluster'
cluster_count = len(np.unique(labeled_finalfeatures_pd[cluster_column]))
cluster_colors = [cm.nipy_spectral(float(i)/cluster_count) for i in range(cluster_count)]

# COMMAND ----------

# MAGIC %md ## Step 2: Profile Segments
# MAGIC 
# MAGIC To get us started, let's revisit the 2-dimensional visualization of our clusters to get us oriented to the clusters.  The color-coding we use in this chart will be applied across our remaining visualizations to make it easier to determine the cluster being explored:

# COMMAND ----------

# DBTITLE 1,Visualize Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=labeled_finalfeatures_pd,
  x='Dim_1',
  y='Dim_2',
  hue=cluster_column,
  palette=cluster_colors,
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md  The segment design we came up with does not produce equal sized groupings.  Instead, we have one group a bit larger than the others, though the smaller groups are still of a size where they are useful to our team:

# COMMAND ----------

# DBTITLE 1,Count Cluster Members
# count members per cluster
cluster_member_counts = labeled_finalfeatures_pd.groupby([cluster_column]).agg({cluster_column:['count']})
cluster_member_counts.columns = cluster_member_counts.columns.droplevel(0)

# plot counts
plt.bar(
  cluster_member_counts.index,
  cluster_member_counts['count'],
  color = cluster_colors,
  tick_label=cluster_member_counts.index
  )

# stretch y-axis
plt.ylim(0,labeled_finalfeatures_pd.shape[0])

# labels
for index, value in zip(cluster_member_counts.index, cluster_member_counts['count']):
    plt.text(index, value, str(value)+'\n', horizontalalignment='center', verticalalignment='baseline')

# COMMAND ----------

# MAGIC %md Let's now examine how each segment differs relative to our base features.  For our categorical features, we'll plot the proportion of cluster members identified as participating in a specific promotional activity relative to the overall number of cluster members. For our continuous features, we will visualize values using a whisker plot:

# COMMAND ----------

# DBTITLE 1,Define Function to Render Plots
def profile_segments_by_features(data, features_to_plot, label_to_plot, label_count, label_colors):
  
    feature_count = len(features_to_plot)
    
    # configure plot layout
    max_cols = 5
    if feature_count > max_cols:
      column_count = max_cols
    else:
      column_count = feature_count      
      
    row_count = math.ceil(feature_count / column_count)

    fig, ax = plt.subplots(row_count, column_count, figsize =(column_count * 4, row_count * 4))
    
    # for each feature (enumerated)
    for k in range(feature_count):

      # determine row & col position
      col = k % column_count
      row = int(k / column_count)
      
      # get axis reference (can be 1- or 2-d)
      try:
        k_ax = ax[row,col]
      except:
        pass
        k_ax = ax[col]
      
      # set plot title
      k_ax.set_title(features_to_plot[k].replace('_',' '), fontsize=7)

      # CATEGORICAL FEATURES
      if features_to_plot[k][:4]=='has_': 

        # calculate members associated with 0/1 categorical values
        x = data.groupby([label_to_plot,features_to_plot[k]]).agg({label_to_plot:['count']})
        x.columns = x.columns.droplevel(0)

        # for each cluster
        for c in range(label_count):

          # get count of cluster members
          c_count = x.loc[c,:].sum()[0]

          # calculate members with value 0
          try:
            c_0 = x.loc[c,0]['count']/c_count
          except:
            c_0 = 0

          # calculate members with value 1
          try:
            c_1 = x.loc[c,1]['count']/c_count
          except:
            c_1 = 0

          # render percent stack bar chart with 1s on bottom and 0s on top
          k_ax.set_ylim(0,1)
          k_ax.bar([c], c_1, color=label_colors[c], edgecolor='white')
          k_ax.bar([c], c_0, bottom=c_1, color=label_colors[c], edgecolor='white', alpha=0.25)


      # CONTINUOUS FEATURES
      else:    

        # get subset of data with entries for this feature
        x = data[
              ~np.isnan(data[features_to_plot[k]])
              ][[label_to_plot,features_to_plot[k]]]

        # get values for each cluster
        p = []
        for c in range(label_count):
          p += [x[x[label_to_plot]==c][features_to_plot[k]].values]

        # plot values
        k_ax.set_ylim(0,1)
        bplot = k_ax.boxplot(
            p, 
            labels=range(label_count),
            patch_artist=True
            )

        # adjust box fill to align with cluster
        for patch, color in zip(bplot['boxes'], label_colors):
          patch.set_alpha(0.75)
          patch.set_edgecolor('black')
          patch.set_facecolor(color)
    

# COMMAND ----------

# DBTITLE 1,Render Plots for All Base Features
# get feature names
feature_names = labeled_basefeatures_pd.drop(label_columns, axis=1).columns

# generate plots
profile_segments_by_features(labeled_basefeatures_pd, feature_names, cluster_column, cluster_count, cluster_colors)

# COMMAND ----------

# MAGIC %md There's a lot to examine in this plot but the easiest thing seems to be to start with the categorical features to identify groups responsive to some promotional offers and not others.  The continuous features then provide a bit more insight into the degree of engagement when that cluster does respond.  
# MAGIC 
# MAGIC As you work your way through the various features, you will likely start to form descriptions of the different clusters.  To assist with that, it might help to retrieve specific subsets of features to focus your attention on a smaller number of features:

# COMMAND ----------

# DBTITLE 1,Plot Subset of Features
feature_names = ['has_pdates_campaign_targeted', 'pdates_campaign_targeted', 'amount_list_with_campaign_targeted']

profile_segments_by_features(labeled_basefeatures_pd, feature_names, cluster_column, cluster_count, cluster_colors)
