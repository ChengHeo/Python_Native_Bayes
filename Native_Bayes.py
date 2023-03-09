#!/usr/bin/env python
# coding: utf-8

# In[1]:


# firstly you need import important packages to read you dataset and run math function
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Data collection

# In[2]:


#read your dataset
df = pd.read_csv("iris.csv")


# # Data Prepocessing

# In[3]:


# To view the first 10 rows of the dataset
df.head(10)


# In[4]:


#check the duplicates in the datatse using the ID "beacsuse its a unique value"
df.duplicated(subset=['Id']).sum()


# In[5]:


# To check the rows and the columns
df.shape


# In[6]:


# Drop the Id column as it will not contributed in the prediction model
df = df.drop(columns= ['Id'])


# In[7]:


# Recheck the number of rows and columns
df.shape


# In[8]:


# To display statistics about the dataset
df.describe()


# In[9]:


# To check about the information and what type of the data we have
df.info()


# In[10]:


# to display the number of samples on each class
df["Species"].value_counts()


# As this is a classification problem, It is necessary to check whether ot not the target feature is skewed i.e.
# imbalanced

# In[11]:


#checking class skewness
plt.figure(figsize=(12, 6))
sns.countplot(x='Species', data=df)
plt.xlabel('Flower Species')
plt.ylabel('Frequency')
plt.title('Count plot of Species', size=16)


# # Data Cleaning

# In[12]:


# check the null values
df.isnull().sum()


# # Explarotry analysis using data viuslization

# In[13]:


g = sns.pairplot(df, hue='Species')
plt.show()


# After graphing the features in a pair plot, it is clear that the relationship between pairs of features of a iris-setosa (in blue) is distinctly different from those of the other two species.
# 
# There is some overlap in the pairwise relationships of the other two species, iris-versicolor (orange) and iris-virginica (green).
# 
# Also, its shown that the Petal Features are giving a better cluster division compared to the Sepal features.
# 
# This is an indication that the Petals can help in better and accurate Predictions over the Sepal. We will check
# that later.

# In[14]:


#boxplot of predictors
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title('Boxplot of Features', size=16)


# From positioning of boxplots, it can be interpreted that Petal width has minimum values and sepal length has highest values.
# 
# From shape of boxplot, the petal length has highest spread i.e. most variance and the sepal width has lest spread i.e. least variance. In terms of skewness.
# 
# Sepal length and sepal width are approximately normal distribution as the median line is in middle.
# 
# Petal length and petal width are left-skewed as the majority part of the box is below the median.
# 
# There are outliers in the Sepal width above 4 and below 2

# In[15]:


# Correlation or heatmap


# Now, when we train any algorithm, the number of features and their correlation plays an important role. 
# 
# If there are features and many of the features are highly correlated, then training an algorithm with all the featues will reduce the accuracy. 
# 
# Thus features selection should be done carefully. This dataset has less featues but still we will see the correlation.

# In[16]:


# First calcuate the correlatio
correlation = df.corr()
correlation


# In[17]:


# Plot the correlation of the dataset 
plt.figure(figsize=(16, 6))
sns.heatmap(correlation, vmin=-1, vmax=1, annot=True)
plt.title('Correlation Coefficient', size=16)


# Overall, all the diagonal have value as 1 because the feature will have perfect positive correlation with itself.
# 
# As this is classification problem, there is no chance to find the correlation of features with target feature but we can find the possibility of multicollinearity.
# 
# Considering the threshold of +/-90, petal length and petal width are highly correlated.

# In[18]:


# use Label Encoder to convert the class varible into numeric format 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[19]:


# after converting the data will be stored again in the class variable
df["Species"] = le.fit_transform(df["Species"])


# In[20]:


# create additional dictionary with mapping 
le_name_mapping = {label: index for index, label in enumerate(le.classes_)}
print(le_name_mapping)


# In[21]:


# Recheck
df.info()


# # Training the Model

# In[22]:


# Before training the model we need train data and test data to validate
from sklearn.model_selection import train_test_split


# In[23]:


# first we need to identify the x and the y
x = df.drop(columns=["Species"])
y = df["Species"]


# we will indicate the target (class label) to be predicted and the columns that will serve as features. Three implementations of Naive Bayes are included in the sklearn library: Gaussian Naive Bayes, Bernoulli Naive Bayes and Multinomial Naive Bayes.

# In[24]:


# split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 0)


# # 1.0 Gaussian Naive Bayes

# Gaussian Naive Bayes is useful when working with features containing continuous values, which probabilities can be modeled using a Gaussian distribution (normal distribution).
# 
# In this lab, we will use a validation procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold cross validation, the training set is split into k smaller sets. The following procedure is followed for each of the k “folds”: A model is trained using k - 1 of the folds as training data The resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy)
# 

# In[25]:


# Import function for k-fold cross validation
from sklearn.model_selection import cross_val_score
# Import the Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
# Create a Gaussian Naive Bayes classifier with default parameters
gnb = GaussianNB()
# Use 10-fold cross validation to perform training and validation on the training set
# Parameter scoring = 'accuracy' will compute accuracy
scores = cross_val_score(gnb, x_train, y_train, cv = 10, scoring = 'accuracy')
# Display the array containing accuracy from 10 folds or iterations
scores


# In[26]:


scores.mean()


# In[27]:


# Print the mean accuracy score
print('Accuracy Validation =',scores.mean())


# In[28]:


# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Running prediction
gnb.fit(x_train, y_train)
# Predict the target for the test dataset
test_predict = gnb.predict(x_test)
# Compute the model accuracy on the development set: How often is the classifier correct?
print("Accuracy Test: ", metrics.accuracy_score(y_test, test_predict))


# In[29]:


# ! pip install statsmodels


# In[30]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[31]:


df.columns


# In[32]:


#Iteration 1
X1 = df[['PetalLengthCm', 'PetalWidthCm', 'SepalLengthCm', 'SepalWidthCm']]
 
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X1.columns
 
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X1.values, i)
 for i in range(len(X1.columns))]
 
print(vif_data)


# In[33]:


vif_data["VIF"]= [variance_inflation_factor(X1.values, i)
 for i in range(len(X1.columns))]
print(vif_data)


# In[34]:


X1 = X1.drop('SepalLengthCm', axis=1)


# Using VIF, All of the features have high correlation in first iteration. As mentioned above, computing VIF must be an iterative approach, we will remove the feature with maximum VIF "SepalLengthCm". you can try to compute the VIF again and recheck the accuracy.

# In[35]:


# split the dataset again using the new X
x1_train, x1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.30, random_state=0)


# In[36]:


# Import function for k-fold cross validation
from sklearn.model_selection import cross_val_score
# Import the Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
# Create a Gaussian Naive Bayes classifier with default parameters
gnb = GaussianNB()
# Use 10-fold cross validation to perform training and validation on the training set
# Parameter scoring = 'accuracy' will compute accuracy
scores = cross_val_score(gnb, x1_train, y1_train, cv = 10, scoring = 'accuracy')
# Display the array containing accuracy from 10 folds or iterations
scores


# In[37]:


print('Accuracy Validation =',scores.mean())


# In[38]:


# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Running prediction
gnb.fit(x1_train, y1_train)
# Predict the target for the test dataset
test_predict = gnb.predict(x1_test)
# Compute the model accuracy on the development set: How often is the classifier correct?
print("Accuracy Test: ", metrics.accuracy_score(y1_test, test_predict))


# After removing the high correlated feature the validation accuracy is increased by approximately 1% and this is a high scoring which is around to 95%, which means making the assumption that the continuous features follow the normal distribution might work well with this dataset.

# In[39]:


df0=x_test.assign(Species = y1_test)


# In[40]:


df1=df0.assign(Prediction = test_predict)
df1


# In[41]:


# Import to csv file
df1.to_csv('prediction1.csv')


# # 2.0 Bernoulli Naive Bayes

# Bernoulli Naive Bayes is suitable to be used when features are binary (represented by 0 or 1), which are modeled using a Bernoulli distrbution. As our dataset contains continuous values, we can first transform all the features to binary values using the binarize parameter, if the binarize parameter is set to none, the input is presumed to
# already consist of binary vectors.
# 

# Another few important points about Bernoulli Naive Bayes:
# 
# i.   Suitable for discrete data
# 
# ii.  Designed for binary/boolean features
# 
# iii. If data is not binary, binarization preprocessing will happen internally
# 
# iv.  Can deal with negative numbers

# In[42]:


# Import the Bernoulli Naive Bayes classifier
from sklearn.naive_bayes import BernoulliNB
# Create a Bernoulli Naive Bayes classifier with default parameters
bnb = BernoulliNB(binarize = 0.0)
# Use 10-fold cross validation to perform training and validation on the training set
scores = cross_val_score(bnb, x_train, y_train, cv = 10, scoring = 'accuracy')
# Display the array containing accuracy from 10 folds or iterations
scores


# In[43]:


# Print the mean accuracy score
print('Accuracy =', scores.mean())


# In[44]:


scores = cross_val_score(bnb, x1_train, y1_train, cv = 10, scoring = 'accuracy')
scores


# In[45]:


print('Accuracy =', scores.mean())


# In[46]:


# Running prediction
bnb.fit(x_train, y_train)
# Predict the target for the test dataset
test_predict = bnb.predict(x_test)
# Compute the model accuracy on the development set: How often is the classifier correct?
print("Accuracy (Test): ", metrics.accuracy_score(y_test, test_predict))


# Validation accuracy is around `37%`, which means making the assumption that the continuous features follow the
# Bernoulli distribution might not work well with this dataset. and the test accuracy of `24%` shows the Bernoulli
# Naive Bayes classifier clearly overfit to the training data.

# # 3.0 Multinomial Naive Bayes
# 

# A multinomial distribution is useful to model feature vectors where each value represents, for example, the
# number of occurrences or frequency counts, which are modeled using a multinomial distribution.

# A few important points about Multinomial Naive Bayes:
# 
# i. Suited for classification of data with discrete features (count data)
# 
# ii. Very useful in text processing
# 
# iii. Each text unit will be converted to vector of word count
# 
# iv. Cannot deal with negative numbers

# In[47]:


# Import the Multinomial Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
# Create a Bernoulli Naive Bayes classifier with default parameters
mnb = MultinomialNB()
# Use 10-fold cross validation to perform training and validation on the training set
scores = cross_val_score(mnb, x_train, y_train, cv = 10, scoring = 'accuracy')
# Display the array containing accuracy from 10 folds or iterations
scores


# In[48]:


# Print the mean accuracy score
print('Accuracy =', scores.mean())


# In[49]:


# Running prediction
mnb.fit(x_train, y_train)
# Predict the target for the test dataset
test_predict = mnb.predict(x_test)
# Compute the model accuracy on the development set: How often is the classifier correct?
print("Accuracy (Test): ", metrics.accuracy_score(y_test, test_predict))


# We can observe a significant drop in accuracy to only 60% using Multinomial Naive Bayes. The features in the
# dataset are not represented by counts so it makes sense that Multinomial Naive Bayes is not a suitable classifier
# for this dataset. Also clear overfitting by observing the test accuracy

# In[ ]:




