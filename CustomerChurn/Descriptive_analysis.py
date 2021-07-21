#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This analysis focuses on the behavior of bank customers who are more likely to leave the bank 
#(i.e. close their bank account).
#I want to find out the most striking behaviors of customers through Exploratory Data Analysis and
#later on use some of the predictive analytics techniques to determine the customers who are most likely to churn.


# In[23]:


'''Descriptive Statistics is the building block of data science. 
In simple terms, descriptive statistics can be defined as the measures that summarize a given data,
and these measures can be broken down further into the measures of central tendency, measures of dispersion and Graphs.

Measures of central tendency include mean, median, and the mode, while the measures of variability include
standard deviation, variance, and the interquartile range. .

I will be explaining:

Mean
Median
Mode
Standard Deviation
Variance
Interquartile Range
Skewness'''


# In[24]:


'''Data set:
CustomerId—contains random values and has no effect on customer leaving the bank.
CreditScore—can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
City—a customer’s location can affect their decision to leave the bank.
Gender—it’s interesting to explore whether gender plays a role in a customer leaving the bank.
Age—this is certainly relevant, since older customers are less likely to leave their bank than younger ones.
BranchId - It is not relevant, all services of bank can be done from branch or online
Tenure—refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
Balance—also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
NumOfProducts—refers to the number of products that a customer has purchased through the bank.
PrimaryAcHolder  - This is the person who is legally responsible for the debt and balance along with the maintenance of the account. 
HasOnlineService - Required for easy and 24/7 service    
HasCrCard—denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.
PrefContact - account holder contact details         
IsActiveMember—active customers are less likely to leave the bank.
EstimatedSalary—as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
Exited—whether or not the customer left the bank.'''


# In[69]:


# Importing Libraries and Chanding directory

import os
import pandas as pd
import numpy as np
import statistics as st 
import seaborn as sns
print(os.getcwd())
os.chdir("C:\\Leina\\Data_sets\\TD_assignment")


# In[55]:


# Load the Data

df = pd.read_csv("hackathon_train_main.csv")
print(df.shape)
print(df.info())


# In[27]:


#Five of the variables are categorical (labelled as 'object') while the remaining are numerical (labelled as 'int' or 'Float').


# In[56]:


#Dropping some irrelavant features for Desriptive Analysis


df.drop(["CustomerId","City","BranchId","PrefLanguage","PrefContact"], axis = 'columns', inplace = True)


# In[57]:


#Measures of Central Tendency
#Measures of central tendency describe the center of the data, and are represented by the mean, the median, and the mode.

df.mean()

round(df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "PrimaryAcHolder", "HasCrCard"
         ,"IsActiveMember", "EstimatedSalary", "Exited"]].mean(),2)
 


# In[31]:


#From the output, we can infer that the average age of the applicant is 40 years, 
#the average balance 76542, average estimated salary is 101700 and the average tenure 5 years.


# In[58]:


#Median
#Median represents the 50th percentile, or the middle value of the data, that separates the distribution into two halves.
#The line of code below prints the median of the numerical variables in the data. 
#The command df.median(axis = 0) will also give the same output.

round(df.median(),2)


# In[ ]:


#From the output, we can infer that the median age of the applicants is 40 years, 
#the median balance is 99681, esitimated salary is 102443 and the median tenure is 5 years. 
#There is a difference between the mean and the median values of these variables, which is because of the distribution of the data.


# In[59]:


#Mode
#Mode represents the most frequent value of a variable in the data.
#This is the only central tendency measure that can be used with categorical variables, 
#unlike the mean and the median which can be used only with quantitative data.

df.mode()


# In[60]:


df.loc[:,"Age"].mode()


# In[ ]:


'''The interpretation of the mode is simple. 
The output above shows that most of the applicants are female, as depicted by the 'Gender'. 
Similar interpreation could be done for the other categorical variables like 'City' and 'PrefLanguage'.
For numerical variables, the mode value represents the value that occurs most frequently. 
For example, the mode value of 40 for the variable 'Age' means that the highest number (or frequency) of applicants are 40 years
old.'''


# In[61]:


#Measures of Dispersion
#We have seen in the data, the values of central tendency measures differ for many variables.
#This is because of the extent to which a distribution is stretched or squeezed. 
#In statistics, this is measured by dispersion which is also referred to as variability, scatter, or spread.
#The most popular measures of dispersion are standard deviation, variance, and the interquartile range.

#Standard Deviation: it is a measure that is used to quantify the amount of variation of a set of data values from its mean. 
#A low standard deviation for a variable indicates that the data points tend to be close to its mean, and vice versa. 
#The line of code below prints the standard deviation of all the numerical variables in the data.

df.std()


# In[36]:


#While interpreting standard deviation values, it is important to understand them in conjunction with the mean.
#For example, in the above output, the standard deviation of the variable 'Balance' is much higher than that of the 
#variable 'CreditScore'. However, the unit of these two variables is different and, therefore, 
#comparing the dispersion of these two variables on the basis of standard deviation alone will be incorrect.
#This needs to be kept in mind.

print(df.loc[:,'Age'].std())
print(df.loc[:,'Balance'].std())

#calculate the standard deviation of the first five rows 
df.std(axis = 1)[0:3]


# In[62]:


#Variance
#Variance is another measure of dispersion.
#It is the square of the standard deviation and the covariance of the random variable with itself.

df.var()


# In[63]:


#Interquartile Range (IQR)
#The Interquartile Range (IQR) is a measure of statistical dispersion,
#and is calculated as the difference between the upper quartile (75th percentile) and the lower quartile (25th percentile).
#The IQR is also a very important measure for identifying outliers and could be visualized using a boxplot.

#IQR can be calculated using the iqr() function. 

from scipy.stats import iqr
iqr(df['Age'])


# In[64]:


import seaborn as sns
import matplotlib.pyplot as plt
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = df, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][0])
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][1])


# In[65]:


#Skewness
#It is the measure of the symmetry, or lack of it, 
#The skewness value can be positive, negative, or undefined. 
#In a perfectly symmetrical distribution, the mean, the median, and the mode will all have the same value.
#However, the variables in our data are not symmetrical, resulting in different values of the central tendency.

print(df.skew())


# In[66]:


#Putting Everything Together
#We have learned the measures of central tendency and dispersion. 
#It is important to analyse these individually, however, because there are certain useful functions in python 
#that can be called upon to find these values. 
#One such important function is the .describe() function that prints the summary statistic of the numerical variables. 


df.describe(include = "all")


# In[72]:


#Correlation Between Features
#Correlation is a statistical term which in common usage refers to how close two variables are to having a 
#linear relationship with each other.
#eatures with high correlation are more linearly dependent and hence have almost the same effect on the dependent variable.
#So, when two features have high correlation, we can drop one of the two features.

plt.figure(figsize=(15, 15
                   ))
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[ ]:




