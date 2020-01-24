#!/usr/bin/env python
# coding: utf-8

# ## Exploratory data anylsis NCS dataset

# Start by importing the dataset and doing some basic data exploration

# In[125]:


import impyute
import csv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from string import ascii_letters
import statsmodels.api as sm
import pickle
# Load libraries
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from impyute.imputation.cs import mice
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


# Join the child table, to add importand variables on the birth to the mother ID number (MOM_PIDX)

# In[126]:


# Variables to join are: CHILD_SEX, CHILD_RACE, GESTATIONAL_AGE, MULTIPLE, SIBLINGS, 
# MOM_RACE, MOM_AGE, MOM_INSURANCE, MOM_MARISTAT, MOM_EDUCATION.


# In[127]:


data= pd.read_csv('ncs_mom.csv')
#data


# In[128]:


data_c= pd.read_csv('ncs_child.csv')
data_c1 = data_c.filter(['MOM_PIDX', 'GESTATIONAL_AGE', 'MOM_AGE', 'MOM_INSURANCE'])
#data_c1


# In[75]:


#df_1 = pd.merge(data, data_c1, join='inner')
#df_1 = pd.merge([data.set_index('MOM_PIDX'), data_c1.set_index('MOM_PIDX'),how ='right']).dropna()
df_1 = data.merge(data_c1, how='right', on='MOM_PIDX').dropna()

#result = pd.concat([df1, df4], axis=1, join='inner')
df_1


# In[129]:


#remove M and switch to NAN
#df_2=[df_1!='M']
df_2= df_1.replace(to_replace=['M'], value=np.NaN)
df_2.isnull().sum()
#df_2


# In[130]:


#for the time being, drop columns with high number of NaN
df_2 = df_2.drop(['BMI','BMI_CAT','HEALTH','GESTATIONAL_AGE', 'MOM_AGE', 'MOM_INSURANCE', 'MOM_PIDX'], axis=1)


# In[131]:


#now drop all Nan across remaining rows
df_2 = df_2.dropna()
df_2.isnull().sum()


# In[132]:


# start the MICE training to impute missing values
#imputed_training=mice(train.values)


# In[13]:


print(df_1['BMI'].describe())


# In[33]:


#min(df1.BMI.values)


# In[133]:


#plot bmi counts for preterm labour
#boxplot(BMI_CAT ~ EARLY_LABOR,xlab="Pre-term labour",ylab="BMI category",col=c("pink","lightblue"),
        #main="Exploratory Data Analysis Plot\n BMI and preterm labour")
#sns.boxplot(data = df1, x='EARLY_LABOR', y='BMI')


# In[134]:


#sns.countplot(data = data, x = 'BMI_CAT')


# In[135]:


print(len(data[data['EARLY_LABOR']==0]))
print(len(data[data['EARLY_LABOR']==1]))


# In[136]:


count_no_con = len(data[data['EARLY_LABOR']==0])
count_yes = len(data[data['EARLY_LABOR']==1])
pct_of_no_con = count_no_con/(count_no_con+count_yes)
print("percentage of term labor", pct_of_no_con*100)
pct_of_yes = count_yes/(count_no_con+count_yes)
print("percentage of preterm labor", pct_of_yes*100)


# In[122]:


unbalanced = sns.countplot(x='EARLY_LABOR',data=data)
#unbalanced.savefig("unbalanced.png")
(unbalanced).get_figure().savefig('unbalanced.png')


# In[137]:


#data.groupby('EARLY_LABOR').mean()


# In[138]:


#data.groupby('BMI_CAT').mean()


# In[139]:


#data.groupby('NAUSEA').mean()


# In[43]:


pd.crosstab(data.HIGHBP_PREG,data.EARLY_LABOR).plot(kind='bar')
plt.title('Early Labor by pregnancy hypertension status')
plt.xlabel('HIGHBP_PREG')
plt.ylabel('EARLY_LABOR')
#to save the figures, use plt.savefig('name')


# In[44]:


data.BMI.hist()
plt.title('Histogram of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.xticks(range(15, 40, 10))


# In[140]:


Var_Corr = df_2.corr()


# In[143]:


# plot the heatmap and annotation on it
heat = sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns)
#heat.savefig("heat.png", bbox_inches='tight', dpi=600)
(heat).get_figure().savefig('heat.png')


# In[144]:


#now we split our sample in a training and test dataset, while balancing the data (for training only)
#first, we make dummy variables for the categorical variables:
df=pd.get_dummies(df_2)


# In[145]:


# Separate majority and minority classes
from sklearn.utils import resample
df_majority = df_2[df_2.EARLY_LABOR == 0]
df_minority = df_2[df_2.EARLY_LABOR == 1]


# In[146]:


# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=430,    # to match the USA prevalence
                                 random_state=100) # reproducible results


# In[147]:


# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])


# In[148]:


# Display new class counts
df_upsampled.EARLY_LABOR.value_counts()


# In[149]:


#now train the balanced dataset:
# Separate input features (X) and target variable (y)
y = df_upsampled.EARLY_LABOR
X = df_upsampled.drop('EARLY_LABOR', axis=1)


# In[167]:


#now we split in training and test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg.fit(X_train, y_train)
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
###########################################


# In[168]:


y_pred = logreg.predict(X_test)


# ### Inizia logistic regression

# In[160]:


#use recursive feature elimination to see if any of the variables are not needed in the model:
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(X, y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


# In[161]:


#apply the model
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# In[162]:


#the only variables with a significant p value for prediction are age, new user and pages visited, drop the others
X = df_upsampled.drop(['URINE','KIDNEY','HIGHBP_PREG','EARLY_LABOR'], axis=1)


# In[163]:


#re-run the model with optimized variables
logit_model=sm.Logit(y,X)


# In[164]:


pkl_filename = 'basic_logreg.pkl'
with open(pkl_filename,'wb') as file:
    pickle.dump(result2,file)


# In[174]:


result2=logit_model.fit()
print(result2.summary2())


# In[166]:


thy = np.exp(-1.3898)
print('The log odds for age varible is:',thy)
#intepretation: Having thyroid disease increases the probability of preterm labor by 25%


# In[169]:


#make a confusion matrix:
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
############################################


# In[ ]:


import plotly.graph_objects as go

fig_model = go.Figure(data=[go.Table(header=dict(values=(classification_report(y_test, y_pred))),
                 cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))
                     ])
fig_model.show()


# In[171]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[172]:


#calculate ROC and make a graph
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
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


age_lo = np.exp(-0.1638)
print('The log odds for age varible is:',age_lo)
#intepretation: an increase of age from website visitors decreses the probability of making a purchase by 15%
user_type_lo = np.exp(-2.0298)
print('The log odds for new user vs repeated user varible is:',user_type_lo)
#intepretation: being a repeated user decreases the probability of a purchase being made by 87% 
pages_lo = np.exp(0.6416)
print('The log odds for number of pages visited varible is:',pages_lo)
#intepretation: for every additional webpage visited in by the user, the probability of making a purchase increases by 89%


# In[86]:


# Train model
clf_1 = LogisticRegression().fit(X, y)


# In[87]:


pred_y_1 = clf_1.predict(X)


# In[88]:


# Is our model still predicting just one class?
print( np.unique( pred_y_1 ) )
 
# How's our accuracy?
print( accuracy_score(y, pred_y_1) )


# In[93]:


# Predict class probabilities
prob_y_2 = clf_1.predict_proba(X)


# In[92]:


# Keep only the positive class
prob_y_2 = [p[1] for p in prob_y_2]


# In[94]:


print(roc_auc_score(y, prob_y_2) )


# In[ ]:




