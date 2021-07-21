#!/usr/bin/env python
# coding: utf-8

# ## Feature Selection
# #### Feature selection is the process of reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model.

# ## F - score
# ________
# 
# #### The F-score, also called the F1-score, is a measure of a model’s accuracy on a dataset. It is used to evaluate binary classification systems, which classify examples into ‘positive’ or ‘negative’.
# #### The F-score is a way of combining the precision and recall of the model, and it is defined as the harmonic mean of the model’s precision and recall.
# ![](https://images.deepai.org/user-content/9954225913-thumb-4901.svg)
# #### The  **chi2** test returns 2 values : **F-score** and **p - value**. Based on the F-score for each feature, we will check the accuracy while considering different number of features for training at a time. Features with high F-score value are of importance.
# ________

# ### Importing the required libraries 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
def split(df,label):
    X_tr, X_te, Y_tr, Y_te = train_test_split(df, label, test_size=0.25, random_state=42)
    return X_tr, X_te, Y_tr, Y_te


from sklearn.feature_selection import chi2

def feat_select(df,f_score_val,num):
    feat_list = list(f_score_val["Feature"][:num])
    return df[feat_list]


from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score

classifiers = ['LinearSVM', 'RadialSVM', 
               'Logistic',  'RandomForest', 
               'AdaBoost',  'DecisionTree', 
               'KNeighbors','GradientBoosting']

models = [svm.SVC(kernel='linear'),
          svm.SVC(kernel='rbf'),
          LogisticRegression(max_iter = 1000),
          RandomForestClassifier(n_estimators=200, random_state=0),
          AdaBoostClassifier(random_state = 0),
          DecisionTreeClassifier(random_state=0),
          KNeighborsClassifier(),
          GradientBoostingClassifier(random_state=0)]


def f_score(df,label):
    chi_values=chi2(df,label)
    score = list(chi_values[0])
    feat = df.columns.tolist()
    fscore_df = pd.DataFrame({"Feature":feat, "Score":score})
    fscore_df.sort_values(by="Score", ascending=False,inplace = True)
    fscore_df.reset_index(drop=True, inplace=True)
    return fscore_df
    
    
def acc_score(df,label):
    Score = pd.DataFrame({"Classifier":classifiers})
    j = 0
    acc = []
    X_train,X_test,Y_train,Y_test = split(df,label)
    for i in models:
        model = i
        model.fit(X_train,Y_train)
        predictions = model.predict(X_test)
        acc.append(accuracy_score(Y_test,predictions))
        j = j+1     
    Score["Accuracy"] = acc
    Score.sort_values(by="Accuracy", ascending=False,inplace = True)
    Score.reset_index(drop=True, inplace=True)
    return Score


def acc_score_num(df,label,f_score_val,feat_list):
    Score = pd.DataFrame({"Classifier":classifiers})
    df2 = None
    for k in range(len(feat_list)):
        df2 = feat_select(df,f_score_val,feat_list[k])
        X_train,X_test,Y_train,Y_test = split(df2,label)
        j = 0
        acc = []
        for i in models:
            model = i
            model.fit(X_train,Y_train)
            predictions = model.predict(X_test)
            acc_val = accuracy_score(Y_test,predictions)
            acc.append(acc_val)
            j = j+1  
        feat = str(feat_list[k])
        Score[feat] = acc
    return Score


def plot2(df,l1,l2,p1,p2,c = "b"):
    feat = []
    feat = df.columns.tolist()
    feat = feat[1:]
    plt.figure(figsize = (16, 18))
    for j in range(0,df.shape[0]):
        value = []
        k = 0
        for i in range(1,len(df.columns.tolist())):
            value.append(df.iloc[j][i])
        plt.subplot(4, 4,j+1)
        ax = sns.pointplot(x=feat, y=value,color = c ,markers=["."])
        plt.text(p1,p2,df.iloc[j][0])
        plt.xticks(rotation=90)
        ax.set(ylim=(l1,l2))
        k = k+1
        
        
def highlight_max(data, color='aquamarine'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else: 
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


# _____
# ### Function Description
# #### 1. split():
# Splits the dataset into training and test set.
# 
# #### 2. feat_select():
# Returns the dataframe with first 'n' features.
# 
# #### 3. f_score():
# Returns the dataframe with the F-score for each feature.
# 
# #### 4. acc_score():
# Returns accuracy for all the classifiers.
# 
# #### 5. acc_score_num():
# Returns accuracy for all the classifiers for the specified number of features.
# 
# #### 6. plot2():
# For plotting the results.
# 
# _____
# 
# ### The following 3 datasets are used:
# 1. Breast Cancer
# 2. Parkinson's Disease
# 3. PCOS
# 
# _____
# 
# ### Plan of action:
# * Looking at dataset (includes a little preprocessing)
# * F-score (Displaying F-score for each feature)
# * Checking Accuracy (comparing accuracies for different number of features) 
# * Visualization (Plotting the graphs)
# ____

# ______________
# # Breast Cancer
# _____________

# ### 1. Looking at dataset

# In[2]:


data_bc = pd.read_csv("C:/Leina/Data_sets/Breast_cancer/data.csv")
label_bc = data_bc["diagnosis"]
label_bc = np.where(label_bc == 'M',1,0)
data_bc.drop(["id","diagnosis","Unnamed: 32"],axis = 1,inplace = True)

print("Breast Cancer dataset:\n",data_bc.shape[0],"Records\n",data_bc.shape[1],"Features")


# In[3]:


display(data_bc.head())
print("All the features in this dataset have continuous values")


# ### 2. F-score

# In[4]:


f_score_bc = f_score(data_bc,label_bc)
f_score_bc


# ### 3. Checking Accuracy

# In[5]:


score1 = acc_score(data_bc,label_bc)
score1


# In[6]:


num_feat1 = list(range(8,26))
classifiers = score1["Classifier"].tolist()
score_bc = acc_score_num(data_bc,label_bc,f_score_bc,num_feat1)
score_bc.style.apply(highlight_max, subset = score_bc.columns[1:], axis=None)


# #### Best Accuracy with all features : RandomForest Classifier - 0.972
# #### Best Accuracy for multiple classifiers for different number of features - 0.979
# #### Here we can only see a slight improvement.

# ### 4. Visualization

# In[7]:


plot2(score_bc,0.90,1,2.5,0.91,c = "gold")


# ______
# # Parkinson's disease
# _______

# ### 1. Looking at dataset

# In[8]:


data_pd = pd.read_csv("C:/Leina/Data_sets/Breast_cancer/Parkinsson disease.csv")
label_pd = data_pd["status"]
data_pd.drop(["status","name"],axis = 1,inplace = True)
#Dropping columns with negative value as it does not work for chi2 test
for i in data_pd.columns:
    neg = data_pd[i]<0
    nsum = neg.sum()
    if nsum > 0:
        data_pd.drop([i],axis = 1,inplace = True)

print("Parkinson's disease dataset:\n",data_pd.shape[0],"Records\n",data_pd.shape[1],"Features")


# In[9]:


display(data_pd.head())
print("All the features in this dataset have continuous values")


# ### 2. F-score

# In[10]:


f_score_pd = f_score(data_pd,label_pd)
f_score_pd


# ### 3. Checking Accuracy

# In[11]:


score3 = acc_score(data_pd,label_pd)
score3


# In[12]:


num_feat3 = list(range(7,21))
classifiers = score3["Classifier"].tolist()
score_pd = acc_score_num(data_pd,label_pd,f_score_pd,num_feat3)
score_pd.style.apply(highlight_max, subset = score_pd.columns[1:], axis=None)


# #### Best Accuracy with all features : RandomForest Classifier - 0.918
# #### Best Accuracy for multiple classifiers for different number of features - 0.918
# #### Here we see no improvement.

# ### 4. Visualization

# In[13]:


plot2(score_pd,0.65,1.0,1,0.7,c = "orange")


# ________
# # PCOS
# ________

# ### 1. Looking at dataset

# In[14]:


data_pcos = pd.read_csv("C:/Leina/Data_sets/Breast_cancer/PCOS_data.csv")
label_pcos = data_pcos["PCOS (Y/N)"]
data_pcos.drop(["Sl. No","Patient File No.","PCOS (Y/N)","Unnamed: 44","II    beta-HCG(mIU/mL)","AMH(ng/mL)"],axis = 1,inplace = True)
data_pcos["Marraige Status (Yrs)"].fillna(data_pcos['Marraige Status (Yrs)'].describe().loc[['50%']][0], inplace = True) 
data_pcos["Fast food (Y/N)"].fillna(1, inplace = True) 

print("PCOS dataset:\n",data_pcos.shape[0],"Records\n",data_pcos.shape[1],"Features")


# In[15]:


display(data_pcos.head())
print("The features in this dataset have both discrete and continuous values")


# ### 2. F-score

# In[16]:


f_score_pcos = f_score(data_pcos,label_pcos)
f_score_pcos


# ### 3. Checking Accuracy

# In[17]:


score4 = acc_score(data_pcos,label_pcos)
score4


# #### Best Accuracy with all features : RandomForest Classifier - 0.889
# #### Best Accuracy for first (12,20,25) features : DecisionTree Classifier - 0.904
# #### Here we can see an improvement of ~1.5%.

# In[18]:


num_feat4 = list(range(12,28))
classifiers = score4["Classifier"].tolist()
score_pcos = acc_score_num(data_pcos,label_pcos,f_score_pcos,num_feat4)
score_pcos.style.apply(highlight_max, subset = score_pcos.columns[1:], axis=None)


# ### 4. Visualization

# In[19]:


plot2(score_pcos,0.3,1.0,1,0.35,c = "limegreen")

