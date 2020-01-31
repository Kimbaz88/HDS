#!/usr/bin/env python
# coding: utf-8

# In[221]:


import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
#import imblearn
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score



df = pd.read_csv('C:\\Users\\kimam\\Desktop\\Insight\\Data\\Combined_final\\dataset.csv')


# In[224]:


#drop the NaN in my outcome variable, leave the rest as the model should ignore them
df = df[pd.notnull(df['GESTATIONAL_AGE_CAT'])]


# In[225]:


#check the number of Nans in the df
df.isnull().sum()
#since the model later will not be happy with MOM_PIDX, lets drop it now
df1 = df.drop(['Unnamed: 0','MOM_PIDX','GESTATIONAL_AGE'], axis=1)


# In[226]:


#check how many of our target variable we have in the combined dataset:
print(len(df1[df1['GESTATIONAL_AGE_CAT']==0]))
print(len(df1[df1['GESTATIONAL_AGE_CAT']==1]))


# In[227]:


#check the proportion of our target variable in the combined dataset:
count_no_con = len(df1[df1['GESTATIONAL_AGE_CAT']==0])
count_yes = len(df1[df1['GESTATIONAL_AGE_CAT']==1])
pct_of_no_con = count_no_con/(count_no_con+count_yes)
print("percentage of term labor", pct_of_no_con*100)
pct_of_yes = count_yes/(count_no_con+count_yes)
print("percentage of preterm labor", pct_of_yes*100)


# In[228]:


plt.figure(figsize=(10,5))
bal_lev = sns.countplot(x='GESTATIONAL_AGE_CAT',data=df1)
plt.xlabel("Preterm labor: 1 = before 37 weeks, 0 = after 37 weeks", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Class difference between preterm and at term labor", fontsize=15)
plt.show()
(bal_lev).get_figure().savefig('bal_lev_data.png')


# In[229]:


#As the imbalance between the classes is now less pronounced than in the original NCS dataset, 
#we can begin modelling


# In[230]:


df1


# ### Downsampling to balance my classes

# In[231]:


y = df1.GESTATIONAL_AGE_CAT
X = df1.drop('GESTATIONAL_AGE_CAT', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)


# In[232]:


# concatenate our training data back together
B = pd.concat([X_train, y_train], axis=1)


# In[233]:


# separate minority and majority classes
nopreterm = B[B.GESTATIONAL_AGE_CAT==0]
preterm = B[B.GESTATIONAL_AGE_CAT==1]


# In[234]:


# downsample majority
nopreterm_downsampled = resample(nopreterm,
                                replace = False, # sample without replacement
                                n_samples = len(preterm), # match minority n
                                random_state = 27) # reproducible results


# In[235]:


# combine minority and downsampled majority
df_down = pd.concat([nopreterm_downsampled, preterm])


# In[236]:


# checking counts
df_down.GESTATIONAL_AGE_CAT.value_counts()


# ### Upsampling to balance my dataset

# In[237]:


# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
y = df1.GESTATIONAL_AGE_CAT
X = df1.drop('GESTATIONAL_AGE_CAT', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)


# In[238]:


# concatenate our training data back together
A = pd.concat([X_train, y_train], axis=1)


# In[239]:


# separate minority and majority classes
nopreterm = A[A.GESTATIONAL_AGE_CAT==0]
preterm = A[A.GESTATIONAL_AGE_CAT==1]


# In[240]:


# upsample minority
preterm_upsampled = resample(preterm,
                          replace=True, # sample with replacement
                          n_samples=len(nopreterm), # match number in majority class
                          random_state=10) # reproducible results


# In[241]:


# combine majority and upsampled minority
df_up = pd.concat([nopreterm, preterm_upsampled])


# In[242]:


df_up.GESTATIONAL_AGE_CAT.value_counts()


# # Model fitting

# ### lightGBM

# In[243]:


#the first two commands are for the upsampled data, the second ones for the downsampled one
#y_train = df_up.GESTATIONAL_AGE_CAT
#X_train = df_up.drop('GESTATIONAL_AGE_CAT', axis=1)
y_train = df_down.GESTATIONAL_AGE_CAT
X_train = df_down.drop('GESTATIONAL_AGE_CAT', axis=1)


# In[244]:


####This is the output of the hyperparameter tuning.
#Training until validation scores don't improve for 30 rounds
#Early stopping, best iteration is:
#[44]	valid's auc: 0.615275	valid's binary_logloss: 0.669674
#Best score reached: 0.5696836847946726 with params: {'colsample_bytree': 0.7567653721452665, 'min_child_samples': 113, 'min_child_weight': 10.0, 'num_leaves': 21, 'reg_alpha': 0.1, 'reg_lambda': 5, 'subsample': 0.8058026565741467} 


# In[245]:


#This is the second algorithm, after hyperparameter tuning
d_train = lgb.Dataset(X_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 18
params['min_data'] = 50
params['max_depth'] = 10
params['usemissing'] = False
params['min_child_samples'] = 113
params['colsample_bytree'] = 0.7567653721452665
params['min_child_weight'] = 10
params['reg_alpha'] = 0.1
params['reg_lambda'] = 5
params['subsample'] = 0.8058026565741467
#params['is_unbalance'] = True
#params['scale_pos_weight'] = 1
clf2 = lgb.train(params, d_train, 100)



#Prediction
y_pred=clf2.predict(X_test)
#convert into binary values
for i in range(0,99):
    if y_pred[i]>=.3:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0

pkl_filename = 'lgbm.pkl'
with open(pkl_filename,'wb') as file:
    pickle.dump(y_pred,file)

#Confusion matrix
#from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
cm



print('Accuracy score: ', accuracy_score(y_test, y_pred.round()))
print('Precision score: ', precision_score(y_test, y_pred.round()))
print('Recall score: ', recall_score(y_test, y_pred.round()))


# The original model (all the above stuff is after YPT)



#import lightgbm as lgb
#start training the algorithm on the prediction
d_train = lgb.Dataset(X_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
params['usemissing'] = False
#params['is_unbalance'] = True
#params['scale_pos_weight'] = 1
clf = lgb.train(params, d_train, 100)


#Prediction
y_pred=clf.predict(X_test)
#convert into binary values
for i in range(0,99):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0



#Confusion matrix
#from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
cm


print('Accuracy score: ', accuracy_score(y_test, y_pred.round()))
print('Precision score: ', precision_score(y_test, y_pred.round()))
print('Recall score: ', recall_score(y_test, y_pred.round()))


y_pred


#calculate ROC and make a graph
lgbm_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_prob(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Light GBM (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ### Hyperparameter tuning

# In[172]:


def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3


# In[185]:


fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}


# In[186]:


#from scipy.stats import randint as sp_randint
#from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
             'learning_rate':[1e-3, 1e-4],
            }


# In[187]:


gs = RandomizedSearchCV(lgb.LGBMClassifier(), param_test, cv=5, n_jobs = -1, n_iter=1000)


# In[188]:


gs.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


# In[ ]:





# In[ ]:





# Now I will use Shap to gain insights from my model (understand why a certain prediction is made)

# In[146]:


# load JS visualization code to notebook
#shap.initjs()


# In[114]:


#explainer = shap.TreeExplainer(d_train)
#shap_values = explainer.shap_values(X, y=y.values)


# In[115]:


#shap_values = shap.TreeExplainer(d_train).shap_values(X_train)
#shap.summary_plot(shap_values, X_train, plot_type="bar")


# In[116]:


# explain the model's predictions using SHAP
#explainer = shap.TreeExplainer(d_train)
#shap_values = explainer.shap_values(X)


# In[117]:


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
#shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])


# Lets try with Lime for the same reason

# In[ ]:





# In[ ]:




