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
import matplotlib.cm as cm

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
# MAGIC # 3. Retrieve Features ⏬

# COMMAND ----------

# retrieve transformed features
data_PCA = spark.table("german_credit_pca").toPandas()
data_PCA.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. K-means Clustering
# MAGIC <div style="font-family: cursive; line-height: 2; font-size:18px">
# MAGIC     📌 <b>Adapt the Databricks Solution Accelerator.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC   📌 Our first attempt at clustering with make use of the K-means algorithm. K-means is a simple, popular algorithm for dividing instances into clusters around a pre-defined number of centroids (cluster centers). The algorithm works by generating an initial set of points within the space to serve as cluster centers. Instances are then associated with the nearest of these points to form a cluster, and the true center of the resulting cluster is re-calculated. The new centroids are then used to re-enlist cluster members, and the process is repeated until a stable solution is generated (or until the maximum number of iterations is exhausted). A quick demonstration run of the algorithm may produce a result as follows:
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Demonstrate Cluster Assignment

# COMMAND ----------

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

# MAGIC %md
# MAGIC ## 4.2 Iterate over Potential Values of K
# MAGIC 
# MAGIC Our initial model run demonstrates the mechanics of generating a K-means clustering solution, but it also demonstrates some of the **shortcomings** of the approach. First, we need to specify the number of clusters. Setting the value incorrectly can force the creation of numerous smaller clusters or just a few larger clusters, neither of which may reflect what we may observe to be the more immediate and natural structure inherent to the data.
# MAGIC 
# MAGIC Second, the results of the algorithm are **highly dependent on the centroids** with which it is initialized. The use of the **K-means++** initialization algorithm addresses some of these problems by better ensuring that initial centroids are dispersed throughout the populated space, but there is still an element of randomness at play in these selections that can have big consequences for our results.
# MAGIC 
# MAGIC To begin working through these challenges, we will generate a large number of model runs over a range of potential cluster counts. For each run, we will calculate the **sum of squared distances** between members and assigned cluster centroids **(inertia)** as well as a secondary metric **(silhouette score)** which provides a combined measure of inter-cluster cohesion and intra-cluster separation (ranging between -1 and 1 with higher values being better). Because of the large number of iterations we will perform, we will distribute this work across our Databricks cluster so that it can be concluded in a timely manner:
# MAGIC 
# MAGIC NOTE We are using a **Spark RDD** as a crude means of exhaustively searching our parameter space in a distributed manner. This is an simple technique frequently used for efficient searches over a defined range of values.

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
    .crossJoin( spark.range(2,10).withColumnRenamed('id','n')) # cluster counts
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
 

# COMMAND ----------

display(results_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Inertia over Cluster Count
# MAGIC Plotting inertia relative to n, i.e. the target number of clusters, we can see that the total sum of squared distances between cluster members and cluster centers decreases as we increase the number of clusters in our solution. Our goal is not to drive inertia to zero (which would be achieved if we made each member the center of its own, 1-member cluster) but instead to identify the point in the curve where the incremental drop in inertia is diminished. In our plot, we might identify this point as occurring somewhere between 3 and 4 clusters.

# COMMAND ----------

colors=color_mix[0:9]
palette=colors

plt.figure(figsize=(10, 8))
ax = sns.lineplot(results_pd.n, results_pd.inertia, palette=colors, marker='o', markersize=10, markeredgewidth=1, markeredgecolor='black')
plt.title('Elbow method', fontweight = 'bold', fontsize = 14)
plt.xlabel('Number of Clusters', fontweight = 'bold', fontsize = 11)
plt.ylabel('Inertia', fontweight = 'bold', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Silhouette Score over Cluster Count
# MAGIC While providing a second perspective, the plot of silhouette scores reinforces the notion that selecting a number of clusters for K-means is a bit subjective. Domain knowledge coupled with inputs from these and similar charts (such as a chart of the Gap statistic) may help point you towards an optimal cluster count but there are no widely-accepted, objective means of determining this value to date.
# MAGIC 
# MAGIC NOTE: We need to be careful to avoid chasing the highest value for the silhouette score in the knee chart. Higher scores can be obtained with higher values of n by simply pushing outliers into trivially small clusters.
# MAGIC 
# MAGIC For our model, we'll go with a value of 3. Looking at the plot of inertia, there appears to be evidence supporting this value. Examining the silhouette scores, the clustering solution appears to be much more stable at this value than at values further down the range. 
# MAGIC 
# MAGIC With a value for n identified, we now need to generate a final cluster design. Given the randomness of the results we obtain from a K-means run (as captured in the widely variable silhouette scores), we might take a best-of-k approach to defining our cluster model. In such an approach, we run through some number of K-means model runs and select the run that delivers the best result as measured by a metric such as silhouette score. To distribute this work, we'll implement a custom function that will allow us to task each worker with finding a best-of-k solution and then take the overall best solution from the results of that work:
# MAGIC 
# MAGIC NOTE: We are again using an RDD to allow us to distribute the work across our cluster. The iterations RDD will hold a value for each iteration to perform. Using mapPartitions() we will determine how many iterations are assigned to a given partition and then force that worker to perform an appropriately configured best-of-k evaluation. Each partition will send back the best model it could discover and then we will take the best from these.

# COMMAND ----------

plt.figure(figsize=(10, 8))
ax = sns.lineplot(results_pd.n, results_pd.silhouette, palette=colors, marker='o', markersize=10, markeredgewidth=1, markeredgecolor='black')
plt.title('Silhouette Score over Cluster Count', fontweight = 'bold', fontsize = 14)
plt.xlabel('Number of Clusters', fontweight = 'bold', fontsize = 11)
plt.ylabel('Silhouette Score', fontweight = 'bold', fontsize = 11)
plt.grid(axis = 'x', alpha = 0.2)
plt.grid(axis = 'y', alpha = 0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 Identify Best of K Model

# COMMAND ----------

total_iterations = 5000
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

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC     📌 We can now visualize our results to get a sense of how the clusters align with the structure of our data. <br>
# MAGIC 
# MAGIC </div>

# COMMAND ----------

bestofk_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.6 Visualise Best of K Clusters

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
ax = sns.scatterplot(bestofk_model.cluster_centers_[:, 0], bestofk_model.cluster_centers_[:, 1], s=150, ax=ax) # add cluster centers
plt.show()
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

def plot_silhouette_chart(features, labels):
  
  n = len(np.unique(labels))
  
  # configure plot area
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(8, 5)

  # configure plots for silhouette scores between -1 and 1
  ax.set_xlim([-0.1, 1])
  ax.set_ylim([0, len(features) + (n + 1) * 10])
  
  # avg silhouette score
  score = silhouette_score(features, labels)

  # compute the silhouette scores for each sample
  sample_silhouette_values = silhouette_samples(features, labels)

  y_lower = 10

  for i in range(n):

      # get and sort members by cluster and score
      ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
      ith_cluster_silhouette_values.sort()

      # size y based on sample count
      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      # pretty up the charts
      color = cm.nipy_spectral(float(i) / n)
      
      ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

      # label the silhouette plots with their cluster numbers at the middle
      ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

      # compute the new y_lower for next plot
      y_lower = y_upper + 10  # 10 for the 0 samples


  ax.set_title("Average silhouette of {0:.3f} with {1} clusters".format(score, n))
  ax.set_xlabel("The silhouette coefficient values")
  ax.set_ylabel("Cluster label")

  # vertical line for average silhouette score of all the values
  ax.axvline(x=score, color="red", linestyle="--")

  ax.set_yticks([])  # clear the yaxis labels / ticks
  ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
  
  return fig, ax

_ = plot_silhouette_chart(data_PCA, bestofk_clusters)

# COMMAND ----------

# MAGIC %md From the silhouette chart, we would appear to have clusters that are roughly balanced, i.e., clusters have roughly the same number of points for cluster centres,k = 3.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Quick Analysis and Profiling

# COMMAND ----------

data_0 = spark.table("german_credit").toPandas()
data_0['cluster'] = bestofk_clusters

# COMMAND ----------

data_0.groupby('cluster').mean()[['Age', 'Job', 'Credit_amount', 'Duration']].round(1)

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="font-family: Cursive; line-height: 2; font-size:18px">
# MAGIC   📌 Cluster 0: Lower credit amount than cluster 2, older age group, shortest duration customers.<br>
# MAGIC   📌 Cluster 1: Lowest credit amount, youger age group, shorter duration customers.<br>
# MAGIC   📌 Cluster 2: Higher credit amount, middle-aged, long duration customers.<br>
# MAGIC </div>
