import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import boxcox1p

house_data=pd.read_csv('processed_train.csv')
test_data=pd.read_csv('processed_test.csv')
train_columns=['bedrooms','bathrooms','no_of_units','area_size',
               'commercial_center','hawker_center','primary_school','secondary_school','shopping_mall','train_station',
               'no_of_poi','tenure_0','tenure_1','tenure_2','ccr','rcr','ocr','type']
train_data=house_data[train_columns]
train_label=house_data['price']

num_train=len(train_data)
test_data=test_data[train_columns]
'''
# Apply box-cox transformation
all_data=pd.concat((train_data,test_data)).reset_index(drop=True)
skewed_feats = all_data[train_columns].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness)
skewness = skewness[abs(skewness) > 0.75]
skewed_features = skewness.index
lam = 1
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
train_data=all_data[:num_train]
test_data=all_data[num_train:]
'''
params = {'max_bin': 256, #max number of bins that feature values will be bucketed in
    'num_leaves': 255,
    'learning_rate': 0.13, # I find out this affects the results a lot
    'boosting': 'goss', #type of boosting tree model: gbdt, rf, dart, goss
    'objective':'regression',
    'max_depth':10,
    'tree_learner': 'data',
          #type of tree learner, 'serial':single machine tree learner
          # 'feature',or 'data' feature/data-parallel tree learner
    'task': 'train',
    'is_training_metric': False, # set this to true to output metric result over training dataset
    'min_data_in_leaf': 1, # These two parameters to avoid error of Check failed
    'min_sum_hessian_in_leaf': 1,
    'bagging_fraction': 1, # This value should be less than 1 if choose 'rf' (random forest)
    'bagging_freq': 2,
    'device': 'gpu',
    'lambda_l1':0.1,
    'lambda_l2': 2
}

Train_Matrix=lgb.Dataset(train_data,label=train_label)
lgb_model=lgb.train(params,Train_Matrix,num_boost_round=100,verbose_eval=0)
results=lgb.cv(params,Train_Matrix,num_boost_round=100,nfold=5,metrics='rmse',shuffle=True,stratified=False,verbose_eval=0)
print(results['rmse-mean'][-1], results['rmse-stdv'][-1])

y_pred=lgb_model.predict(test_data)

result_df=pd.DataFrame()
result_df['Id']=pd.Series(range(len(y_pred)))
result_df['Predicted']=pd.Series(y_pred)
result_df.to_csv('submission.csv',index=False)
