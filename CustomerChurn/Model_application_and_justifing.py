#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print(os.getcwd())
os.chdir("C:\\Leina\\Data_sets\\TD_assignment")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import  confusion_matrix , plot_roc_curve, classification_report


# In[6]:


# Read Data

train = pd.read_csv("hackathon_train_main.csv")

train.head()


# In[7]:


train.shape


# In[8]:


train.columns


# In[9]:


#Exploratory Data Analysis

#Below are the steps involved to understand, clean and prepare your data for building your predictive model:

# Variable Identification
# Univariate Analysis
# Bi-variate Analysis
# Missing values treatment
# Outlier treatment
# Variable transformation
# Variable creation


# In[10]:


# Missing Data Analysis

train.isnull().sum()


# In[43]:


train.describe


# In[11]:


# Data Type Analysis

train.dtypes


# In[26]:


# Univariate Analysis
"""At this stage, we explore variables one by one. Method to perform uni-variate analysis will depend on whether the variable 
type is categorical or continuous. 
Let’s look at these methods and statistical measures for categorical and continuous variables individually:

Continuous Variables:- In case of continuous variables, we need to understand the central tendency and spread of the variable.
These are measured using various statistical metrics such as Histogram and Bar plots:"""


# In[12]:


# remove columns which will not help in predicting churn
train.drop(["CustomerId","PrefLanguage","BranchId","CurrencyCode"], axis = 1, inplace = True)


# In[48]:


labels = 'Exited', 'Retained'
sizes = [train.Exited[train['Exited']==1].count(), train.Exited[train['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show()


# In[ ]:


So about 42.8% of the customers have churned. So the baseline model could be to predict that 42.8% of the customers will churn.
Given 42.8% is a high number.


# In[13]:


# We first review the 'Status' relation with categorical variables
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='City', hue = 'Exited',data = train, ax=axarr[0][0])
sns.countplot(x='Gender', hue = 'Exited',data = train, ax=axarr[0][1])
sns.countplot(x='HasCrCard', hue = 'Exited',data = train, ax=axarr[1][0])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = train, ax=axarr[1][1])


# In[ ]:


We note the following:

Majority of the data is from Toronto. 
The proportion of male customers churning is also greater than that of female customers
Interestingly, majority of the customers that churned are those with credit cards. Given that majority of the customers
have credit cards could prove this to be just a coincidence.
Unsurprisingly the active members have a greater churn. 
Worryingly is that the overall proportion of inactive members is quite high suggesting that the bank may need a program 
implemented to turn this group to active customers as this will definately have a positive impact on the customer churn.


# In[55]:


# Relations based on the continuous data attributes
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = train, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = train , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = train, ax=axarr[1][0])
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = train, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = train, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = train, ax=axarr[2][1])


# In[ ]:


We note the following:

There is no significant difference in the credit score distribution between retained and churned customers.

The older customers are churning at more than the younger ones alluding to a difference in service preference
in the age categories. The bank may need to review their target market or review the strategy for retention between
the different age groups

With regard to the tenure, the clients on either extreme end (spent little time with the bank or a lot of time with the bank) are more likely to churn compared to those that are of average tenure.
Worryingly, the bank is losing customers with significant bank balances which is likely to hit their available capital for 
lending.

Neither the product nor the salary has a significant effect on the likelihood to churn.


# In[57]:


#Feature engineering
#We seek to add features that are likely to have an impact on the probability of churning. We first split the train and test sets

# Split Train, test data
df_train = train.sample(frac=0.8,random_state=200)
df_test = train.drop(df_train.index)
print(len(df_train))
print(len(df_test))


# In[58]:


df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df_train)
plt.ylim(-1, 5)


# In[ ]:


#we have seen that the salary has little effect on the chance of a customer churning. 
#However as seen above, the ratio of the bank balance and the estimated salary indicates that customers 
#with a higher balance salary ratio churn more which would be worrying to the bank as this impacts their 
#source of loan capital.


# In[59]:


# Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = df_train)
plt.ylim(-1, 1)
plt.show()


# In[60]:


'''Lastly we introduce a variable to capture credit score given age to take into account credit behaviour visavis adult life
:-)'''
df_train['CreditScoreGivenAge'] = df_train.CreditScore/(df_train.Age)


# In[61]:


# Resulting Data Frame
df_train.head()


# In[62]:


#Data prep for model fitting
# Arrange columns by data type for easier manipulation

continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember','City', 'Gender']
df_train = df_train[['Exited'] + continuous_vars + cat_vars]
df_train.head()


# In[63]:


'''For the one hot variables, we change 0 to -1 so that the models can capture a negative relation 
where the attribute in inapplicable instead of 0'''
df_train.loc[df_train.HasCrCard == 0, 'HasCrCard'] = -1
df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] = -1
df_train.head()


# In[64]:


# One hot encode the categorical variables
lst = ['City', 'Gender']
remove = list()
for i in lst:
    if (df_train[i].dtype == np.str or df_train[i].dtype == np.object):
        for j in df_train[i].unique():
            df_train[i+'_'+j] = np.where(df_train[i] == j,1,-1)
        remove.append(i)
df_train = df_train.drop(remove, axis=1)
df_train.head()


# In[65]:


# minMax scaling the continuous variables
minVec = df_train[continuous_vars].min().copy()
maxVec = df_train[continuous_vars].max().copy()
df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)
df_train.head()


# In[68]:


# data prep pipeline for test data

def DfPrepPipeline(df_predict,df_train_Cols,minVec,maxVec):
    
    # Add new features
    df_predict['BalanceSalaryRatio'] = df_predict.Balance/df_predict.EstimatedSalary
    df_predict['TenureByAge'] = df_predict.Tenure/(df_predict.Age - 18)
    df_predict['CreditScoreGivenAge'] = df_predict.CreditScore/(df_predict.Age - 18)
    
    # Reorder the columns
    continuous_vars = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
    cat_vars = ['HasCrCard','IsActiveMember',"City", "Gender"] 
    df_predict = df_predict[['Exited'] + continuous_vars + cat_vars]
    
    # Change the 0 in categorical variables to -1
    df_predict.loc[df_predict.HasCrCard == 0, 'HasCrCard'] = -1
    df_predict.loc[df_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
    
    # One hot encode the categorical variables
    lst = ["City", "Gender"]
    remove = list()
    for i in lst:
        for j in df_predict[i].unique():
            df_predict[i+'_'+j] = np.where(df_predict[i] == j,1,-1)
        remove.append(i)
    df_predict = df_predict.drop(remove, axis=1)
    
    # Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data
    L = list(set(df_train_Cols) - set(df_predict.columns))
    for l in L:
        df_predict[str(l)] = -1        
    
    # MinMax scaling coontinuous variables based on min and max from the train data
    df_predict[continuous_vars] = (df_predict[continuous_vars]-minVec)/(maxVec-minVec)
    # Ensure that The variables are ordered in the same way as was ordered in the train set
    df_predict = df_predict[df_train_Cols]
    return df_predict


# In[86]:


'''Model fitting and selection
For the model fitting, I will try out the following

Logistic regression in the primal space and with different kernels
SVM in the primal and with different Kernels
Ensemble models'''

# Support functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Scoring functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[71]:


# Function to give best model score and parameters
def best_model(model):
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
def get_auc_scores(y_actual, method,method2):
    auc_score = roc_auc_score(y_actual, method); 
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2); 
    return (auc_score, fpr_df, tpr_df)


# In[73]:


# Fit primal logistic regression
log_primal = LogisticRegression(C=100, class_weight=None, dual=False, 
                                fit_intercept=True,intercept_scaling=1, max_iter=250, multi_class='auto',n_jobs=None, 
                                penalty='l2', random_state=None, solver='lbfgs',tol=1e-05, verbose=0, warm_start=False)
log_primal.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)


# In[74]:


# Fit logistic regression with pol 2 kernel
poly2 = PolynomialFeatures(degree=2)
df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
log_pol2 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=300, 
                              multi_class='auto', n_jobs=None, 
                              penalty='l2', random_state=None, solver='liblinear',tol=0.0001, verbose=0, warm_start=False)
log_pol2.fit(df_train_pol2,df_train.Exited)


# In[76]:


#Fit SVM with RBF Kernel
SVM_RBF = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, 
              kernel='rbf', max_iter=-1, probability=True, 
              random_state=None, shrinking=True,tol=0.001, verbose=False)
SVM_RBF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)


# In[78]:


# Fit SVM with Pol Kernel
SVM_POL = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,  decision_function_shape='ovr', degree=2, gamma=0.1, 
              kernel='poly',  max_iter=-1,
              probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
SVM_POL.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)


# In[79]:


# Fit Random Forest classifier
RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, 
                            max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_impurity_split=None,min_samples_leaf=1, 
                            min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
RF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)


# In[88]:


# Fit Extreme Gradient Boost Classifier
XGB = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
XGB.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
#n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0


# In[ ]:


#Review best model fit accuracy : Keen interest is on the performance in predicting 1's (Customers who churn)


# In[100]:


print(confusion_matrix(df_train.Exited, log_primal.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited, log_primal.predict(df_train.loc[:, df_train.columns != 'Exited'])))


# In[101]:


print(confusion_matrix(df_train.Exited,  log_pol2.predict(df_train_pol2)))
print(classification_report(df_train.Exited,  log_pol2.predict(df_train_pol2)))


# In[102]:


print(confusion_matrix(df_train.Exited,  SVM_RBF.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited,  SVM_RBF.predict(df_train.loc[:, df_train.columns != 'Exited'])))


# In[103]:


print(confusion_matrix(df_train.Exited,  SVM_POL.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited,  SVM_POL.predict(df_train.loc[:, df_train.columns != 'Exited'])))


# In[104]:


print(confusion_matrix(df_train.Exited,  RF.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited,  RF.predict(df_train.loc[:, df_train.columns != 'Exited'])))


# In[97]:


print("Confusion Matrix:")
print(confusion_matrix(df_train.Exited,  XGB.predict(df_train.loc[:, df_train.columns != 'Exited'])))

print(classification_report(df_train.Exited,  XGB.predict(df_train.loc[:, df_train.columns != 'Exited'])))


# In[95]:


y = df_train.Exited
X = df_train.loc[:, df_train.columns != 'Exited']
X_pol2 = df_train_pol2
auc_log_primal, fpr_log_primal, tpr_log_primal = get_auc_scores(y, log_primal.predict(X),log_primal.predict_proba(X)[:,1])
auc_log_pol2, fpr_log_pol2, tpr_log_pol2 = get_auc_scores(y, log_pol2.predict(X_pol2),log_pol2.predict_proba(X_pol2)[:,1])
auc_SVM_RBF, fpr_SVM_RBF, tpr_SVM_RBF = get_auc_scores(y, SVM_RBF.predict(X),SVM_RBF.predict_proba(X)[:,1])
auc_SVM_POL, fpr_SVM_POL, tpr_SVM_POL = get_auc_scores(y, SVM_POL.predict(X),SVM_POL.predict_proba(X)[:,1])
auc_RF, fpr_RF, tpr_RF = get_auc_scores(y, RF.predict(X),RF.predict_proba(X)[:,1])
auc_XGB, fpr_XGB, tpr_XGB = get_auc_scores(y, XGB.predict(X),XGB.predict_proba(X)[:,1])


# In[99]:


plt.figure(figsize = (12,6), linewidth= 1)
plt.plot(fpr_log_primal, tpr_log_primal, label = 'log primal Score: ' + str(round(auc_log_primal, 5)))
plt.plot(fpr_log_pol2, tpr_log_pol2, label = 'log pol2 score: ' + str(round(auc_log_pol2, 5)))
plt.plot(fpr_SVM_RBF, tpr_SVM_RBF, label = 'SVM RBF Score: ' + str(round(auc_SVM_RBF, 5)))
plt.plot(fpr_SVM_POL, tpr_SVM_POL, label = 'SVM POL Score: ' + str(round(auc_SVM_POL, 5)))
plt.plot(fpr_RF, tpr_RF, label = 'RF score: ' + str(round(auc_RF, 5)))
plt.plot(fpr_XGB, tpr_XGB, label = 'XGB score: ' + str(round(auc_XGB, 5)))
plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
#plt.savefig('roc_results_ratios.png')
plt.show()


# In[ ]:


'''From the above results, my main aim is to predict the customers that will possibly churn so they can be put in some sort 
of scheme/promotion of $200 to prevent churn hence the recall measures on the 
1's is of more importance to me than the overall accuracy score of the model.

Given that in the data we had 48% of churn, a recall greater than this baseline will already be an improvement but we
want to get as high as possible while trying to maintain a high precision so that the bank can train its resources effectively 
towards clients highlighted by the model without wasting too much resources on the false positives.

From the review of the fitted models above, the best model that gives a decent balance of the recall and precision is the 
random forest where according to the fit on the training set, with a precision score on 1's of 0.81, 
out of all customers that the model thinks will churn, 81% do actually churn and with the 
recall score of 0.75 on the 1's, the model is able to highlight 74% of all those who churned.


# In[105]:


#Test model prediction accuracy on test data
# Make the data transformation for test data
df_test = DfPrepPipeline(df_test,df_train.columns,minVec,maxVec)
df_test = df_test.mask(np.isinf(df_test))
df_test = df_test.dropna()
df_test.shape


# In[106]:


print(confusion_matrix(df_test.Exited,  RF.predict(df_test.loc[:, df_test.columns != 'Exited'])))

print(classification_report(df_test.Exited,  RF.predict(df_test.loc[:, df_test.columns != 'Exited'])))


# In[107]:


auc_RF_test, fpr_RF_test, tpr_RF_test = get_auc_scores(df_test.Exited, RF.predict
                                                       (df_test.loc[:, df_test.columns != 'Exited']),
                                                       RF.predict_proba(df_test.loc[:, df_test.columns != 'Exited'])[:,1])
plt.figure(figsize = (12,6), linewidth= 1)
plt.plot(fpr_RF_test, tpr_RF_test, label = 'RF score: ' + str(round(auc_RF_test, 5)))
plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
#plt.savefig('roc_results_ratios.png')
plt.show()


# In[15]:


'''The precision of the model on previousy unseen test data is slightly higher with regard to predicting 
1's i.e. those customers that churn. However, in as much as the model has a high accuracy, 
it still some half of those who end up churning.
This could be imporved by providing retraining the model with more data over time.'''


# In[ ]:




