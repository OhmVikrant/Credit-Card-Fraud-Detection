#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# 
# 
# Using a dataset of of nearly 28,500 credit card transactions and multiple unsupervised anomaly detection algorithms, we are going to identify transactions with a high probability of being credit card fraud. In this project, we shall build and deploy the following two machine learning algorithms:
# 
# 1. Local Outlier Factor (LOF)
# 
# 2. Isolation Forest Algorithm
# 
# Furthermore, we shall use metrics suchs as precision, recall, and F1-scores.
# 
# In addition, we shall explore parameter histograms and correlation matrices.

# In[1]:


#importing necessary libraries

import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))


# In[2]:


# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')


# In[4]:


# Start exploring the dataset
print(data.columns)
print(data.shape)


# In[5]:


# Print the shape of the data
data = data.sample(frac=0.1, random_state = 1)      #since original dataset is too large
print(data.shape)
print(data.describe())

# V1 - V28 are the results of a PCA Dimensionality reduction to protect user identities and sensitive features


# In[6]:


# Plot histograms of each parameter 
data.hist(figsize = (20, 20))
plt.show()


# In[7]:


# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(Fraud)))
print('Valid Transactions: {}'.format(len(Valid)))


# In[8]:


# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[9]:


# Get all the columns from the dataFrame
columns = data.columns.tolist()

# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

# Store the variable we'll be predicting on
target = "Class"

X = data[columns]
Y = data[target]

# Print shapes
print(X.shape)
print(Y.shape)


# # The Algorithms

# ### Local Outlier Factor (LOF)
# The anomaly score of each sample is called Local Outlier Factor. It measures the local deviation of density of a given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood.
# 
# ### Isolation Forest Algorithm
# The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
# 
# Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.
# 
# This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
# 
# Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.
# 
# 

# In[10]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest               #anomaly detection
from sklearn.neighbors import LocalOutlierFactor           #anomaly detection




#localOutlierFactor is an unsupervised outlier detection method 
#It calculates the anomaly score of each sample and we call it the localOutlierFactor
#It measures the local deviation of density of a given sample wrt it's neighbors
#It is local and that anomaly score depends on how isolated the object is wrt the surrounding neighborhood
#It is determined in the same way as Kneighbors method




#Isolation Forest explicitly identifies anomalies instead of profiling normal data points
#Isolation Forest, like any tree ensemble method, is built on the basis of decision trees
#In these trees, partitions are created by first randomly selecting a feature and then 
#selecting a random split value between the minimum and maximum value of the selected feature
#In principle, outliers are less frequent than regular observations and are different from them in terms of values 
#they lie further away from the regular observations in the feature space




#define a random state
state = 1

#define the outlier detection methods
#putting into a dictionary of classifiers
classifiers = {
    
    "Isolation Forest": IsolationForest(
                                    max_samples = len(X),
                                    contamination = outlier_fraction,
                                    random_state = state),
    
    "Local Outlier Factor": LocalOutlierFactor(
                                    n_neighbors = 20,
                                    contamination = outlier_fraction)
    }


# In[14]:


# Fit the model
#plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

