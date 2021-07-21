#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm, datasets
iris = datasets.load_iris()


# In[2]:


import pandas as pd
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])
df[47:150]


# In[6]:


#Traditional Method of Train and Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)


# In[7]:


# randomly initialize these parameters, at this point i am not confident about best parameters to choose
# score changes every time with different samples
model = svm.SVC(kernel='rbf',C=30,gamma='auto')
model.fit(X_train,y_train)
model.score(X_test, y_test)


# In[10]:


# finding optimal value for parameters
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],
    'kernel': ['rbf','linear']
}, cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
clf.cv_results_


# In[11]:


# Results above are not easy to view, so lets import these results into dataframe
df = pd.DataFrame(clf.cv_results_)
df


# In[12]:


#looking for only parameter vales and mean score
df[['param_C','param_kernel','mean_test_score']]


# In[13]:


#Finding best parameter combination
clf.best_params_


# In[14]:


clf.best_score_


# In[18]:


dir(clf)


# In[ ]:


# We have tested only 3 value of c, limited parametrs...Can be hard if we test c as a range(computation cost will increase)


# In[15]:


# Use RandomizedSearchCV to reduce number of iterations and with random combination of parameters. 
#This is useful when you have too many parameters to try and your training time is longer.
#It helps reduce the cost of computation

from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
        'C': [1,10,20],
        'kernel': ['rbf','linear']
    }, 
    cv=5, 
    return_train_score=False, 
    n_iter=2
)
rs.fit(iris.data, iris.target)
pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]


# In[16]:


# Dictionary with classifiers and parametrs
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}


# In[17]:


scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[ ]:


#Based on above, I can conclude that SVM with C=1 and kernel='rbf' is the best model for s
#olving my problem of iris flower classification

