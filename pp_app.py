import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle


df = pd.read_csv('C:\\Users\\kimam\\Desktop\\Insight\\Data\\Combined_final\\downsample.csv')
df = df.drop(['Unnamed: 0'], axis= 1)

# Splitting the dataset into the Training set and Test set
y = df.GESTATIONAL_AGE_CAT
X = df.drop('GESTATIONAL_AGE_CAT', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

d_train = lgb.Dataset(X_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'recall'
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

lgbm = lgb.train(params, d_train, 100)

pred_y = lgbm.predict(X_train)

pickle_out = open("lgbm.pkl","wb")
pickle.dump(lgbm, pickle_out)
pickle_out.close()#This is the pickle command needed to store the model for the project app
#pkl_filename = 'lgbm.pkl'
#with open(pkl_filename,'wb') as file:
    #pickle.dump(lgbm,file)

