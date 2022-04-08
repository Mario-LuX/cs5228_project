import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.special import boxcox1p

def district():
    # Get the data to feed the model
    # Using the version of one-hot 'district' and abandon 'ocr','ccr','rcr'
    train_data=pd.read_csv('processed_train.csv')
    test_data=pd.read_csv('processed_test.csv')
    train_data=pd.get_dummies(train_data,columns=['district'])
    test_data=pd.get_dummies(test_data,columns=['district'])
    train_label=train_data['price']
    test_data.drop(columns=['planning_area', 'subszone', 'region','ocr','ccr','rcr'], inplace=True)
    train_data.drop(columns=['planning_area', 'subszone', 'region', 'price','ocr','ccr','rcr'], inplace=True)
    return train_data, train_label, test_data

def normal():
    # Get the data to feed the model
    # Just use the data after preprocessing (i.e. use 'market_segment' division and abandon 'district' etc)
    train_data=pd.read_csv('processed_train.csv')
    test_data=pd.read_csv('processed_test.csv')

    train_label=train_data['price']
    test_data.drop(columns=['district','planning_area','subszone','region'],inplace=True)
    train_data.drop(columns=['district','planning_area','subszone','region','price'],inplace=True)
    return train_data, train_label, test_data

def process_coordinate(train_data, test_data, handle='norm'):
    if handle=='norm':
        train_data['lng'] = (train_data['lng'] - train_data['lng'].mean()) / train_data['lng'].std()
        train_data['lat'] = (train_data['lat'] - train_data['lat'].mean()) / train_data['lat'].std()
        test_data['lng'] = (test_data['lng'] - test_data['lng'].mean()) / test_data['lng'].std()
        test_data['lat'] = (test_data['lat'] - test_data['lat'].mean()) / test_data['lat'].std()
    else:
        train_data['lat'] = (train_data['lat'] - train_data['lat'].min()) / (
                train_data['lat'].max() - train_data['lat'].min())
        train_data['lng'] = (train_data['lng'] - train_data['lng'].min()) / (
                    train_data['lng'].max() - train_data['lng'].min())
        test_data['lat'] = (test_data['lat'] - test_data['lat'].min()) / (
                    test_data['lat'].max() - test_data['lat'].min())
        test_data['lng']=(test_data['lng']-test_data['lng'].min())/(test_data['lng'].max()-test_data['lng'].min())
    return train_data, test_data

def box_cox(train_data,test_data):
    num_train = len(train_data)
    # Apply box-cox transformation
    all_data=pd.concat((train_data,test_data)).reset_index(drop=True)
    skewed_feats = all_data.apply(lambda x: stats.skew(x)).sort_values(ascending=False)
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
    return train_data, test_data


params={
        'booster': 'gbtree', # Type of booster: gbtree, gblinear or dart, default is gbtree
        'rate_drop': 0.1, # only for dart, droppout rate
        'skip_drop':0.5, # only for dart, probability of skipping a pruning operation
        'learning_rate':0.1,
        'min_split_loss': 0.1, #minimum loss reduction to make a further partition
        'max_depth': 10,
        'min_child_weight': 1.5, #minimum sum of instance weight(hessian) needed in a child
        'subsample': 1, #smapling ratio of records in every boosting iteration
        'colsample_bytree': 1, #sampling ratio of features in every boosting iteration
        'lambda': 10, #L2-regularization term
        'alpha': 1, #L1-regularization term
        'tree_method': 'gpu_hist', # use GPU option
        'objective': 'reg:squarederror', # Specify what kind of task and evaluation metrics
        'eval_metric': 'rmse', # Evaluation metric for eval set
    }

def write_submission(y_pred):
    # Write a csv containing the predicted result
    result_df = pd.DataFrame()
    result_df['Id'] = pd.Series(range(len(y_pred)))
    result_df['Predicted'] = pd.Series(y_pred)
    result_df.to_csv('submission.csv', index=False)

if __name__=='__main__':
    train_data, train_label, test_data= district()
    train_data, test_data = process_coordinate(train_data, test_data, handle='min_max')
    # train_data, test_data =box_cox(train_data, test_data)
    Train_Matrix=xgb.DMatrix(train_data,label=train_label)
    Test_Matrix=xgb.DMatrix(test_data)

    model=xgb.train(params,Train_Matrix,num_boost_round=200,evals=[(Train_Matrix,'train')],verbose_eval=20)
    y_pred=model.predict(Test_Matrix)

    results = xgb.cv(params, Train_Matrix, num_boost_round=200, nfold=5, metrics='rmse', verbose_eval=20)
    write_submission(y_pred)


