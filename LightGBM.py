import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import boxcox1p
from sklearn.preprocessing import RobustScaler

def add_crawler(district=True):
    # If using this functin, it means we add more data obtained from crawler into the original training set
    orignal_data=pd.read_csv('processed_train.csv')
    crawler_data=pd.read_csv('train_crawler.csv')
    test_data=pd.read_csv('processed_test.csv')
    feature_chosen=['type','bedrooms','bathrooms','district','lat','lng','no_of_units','area_size','tenure_0','tenure_1','tenure_2',
                    'commercial_center','hawker_center','primary_school','secondary_school','shopping_mall','train_station','no_of_poi']
    label_1=orignal_data['price']
    label_2=crawler_data['price']
    train_label=pd.concat([label_1,label_2],ignore_index=True)
    orignal_data=orignal_data[feature_chosen]
    crawler_data=crawler_data[feature_chosen]
    train_data=pd.concat([orignal_data, crawler_data],ignore_index=True)
    test_data=test_data[feature_chosen]
    if district:
        train_data = pd.get_dummies(train_data, columns=['district'])
        test_data = pd.get_dummies(test_data, columns=['district'])
    else:
        train_data.drop(columns=['district'],inplace=True)
        test_data.drop(columns=['district'],inplace=True)
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    return train_data, train_label, test_data

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

def box_cox(train_data,test_data):
    num_train = len(train_data)
    # Apply box-cox transformation to the training data
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

params = {'max_bin': 255, #max number of bins that feature values will be bucketed in
    'num_leaves': 255,
    'learning_rate': 0.13, # I find out this affects the results a lot
    'boosting': 'goss', #type of boosting tree model: gbdt, rf, dart, goss
    'drop_rate': 0.1, # only for dart, dropout rate
    'skip_drop': 0.5, # only for dart, probability of skipping a pruning operation
    'objective':'regression',
    'max_depth':15,
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
    'lambda_l1': 0.1,
    'lambda_l2': 2
}

"""
A relatively good
    'max_bin': 256, #max number of bins that feature values will be bucketed in
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
    'lambda_l1': 0.1,
    'lambda_l2': 2
"""

def write_submission(y_pred):
    result_df = pd.DataFrame()
    result_df['Id'] = pd.Series(range(len(y_pred)))
    result_df['Predicted'] = pd.Series(y_pred)
    result_df.to_csv('submission.csv', index=False)

def plot_losses():
    # This function is to visualize the difference before and after adding crawled data
    # Need prepared csv file containing loss of each boosting iteration

    loss_df=pd.read_csv('loss.csv')
    crawl_df=pd.read_csv('loss_crawl.csv')
    train_loss_1=np.array(loss_df['Train_loss'])
    cv_loss_1=np.array(loss_df['CV_loss'])
    train_loss_2=np.array(crawl_df['Train_loss'])
    cv_loss_2=np.array(crawl_df['CV_loss'])
    plt.plot(train_loss_1[:200])
    plt.plot(cv_loss_1[:200])
    plt.plot(train_loss_2)
    plt.plot(cv_loss_2)
    plt.title("Train loss and CV loss through iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss(RMSE)")
    plt.legend(["Train loss (Orignial)", "CV loss (Original)","Train loss (Crawler)", "CV loss (Crawler)"])
    plt.show()


if __name__=='__main__':
    # train_data, train_label, test_data = normal()
    train_data, train_label, test_data=add_crawler(district=False)
    # train_data, test_data=process_coordinate(train_data,test_data,handle='norm')
    # transformer=RobustScaler().fit(train_data)
    # train_data=transformer.transform(train_data)
    # train_data, test_data = box_cox(train_data, test_data)
    num_round=200
    Train_Matrix=lgb.Dataset(train_data,label=train_label)
    lgb_model=lgb.train(params,Train_Matrix,num_boost_round=num_round,verbose_eval=0)
    results=lgb.cv(params,Train_Matrix,num_boost_round=num_round,nfold=5,metrics='rmse',
                   shuffle=True,stratified=False,verbose_eval=0,eval_train_metric=True)
    for i in range(num_round):
        if (i+1)%20==0:
            print(i+1, "Train: ",results['train rmse-mean'][i], results['train rmse-stdv'][i], "Test: ",
                  results['valid rmse-mean'][i], results['valid rmse-stdv'][i])

    loss_df=pd.DataFrame()
    loss_df['Train_loss']=pd.Series(results['train rmse-mean'])
    loss_df['CV_loss']=pd.Series(results['valid rmse-mean'])
    loss_df.to_csv('loss_crawl.csv',index=False)
    # save training and cv loss for comparison and plotting
    y_pred=lgb_model.predict(test_data)
    write_submission(y_pred)
    plot_losses()