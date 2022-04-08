import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,  HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
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

xgboost_params={
        'booster': 'gbtree', # Type of booster: gbtree, gblinear or dart, default is gbtree
        'learning_rate':0.2,
        'min_split_loss': 0.1, #minimum loss reduction to make a further partition
        'max_depth': 10,
        'min_child_weight': 1.5, #minimum sum of instance weight(hessian) needed in a child
        'subsample': 1, #smapling ratio of records in every boosting iteration
        'colsample_bytree': 1, #sampling ratio of features in every boosting iteration
        'lambda': 0.1, #L2-regularization term
        'alpha': 0.3, #L1-regularization term
        'tree_method': 'gpu_hist', # use GPU option
        'objective': 'reg:squarederror', # Specify what kind of task and evaluation metrics
        'eval_metric': 'rmse', # Evaluation metric for eval set
    }

lightGBM_params={'max_bin': 256, #max number of bins that feature values will be bucketed in
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

def rmse_cv(model, train ,y_train, n_folds=5):
    kf = KFold(n_folds, shuffle=True).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.1)) #bad model rmse-mean 186w

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5)) # bad model rmse-mean 186w

rf= make_pipeline(RobustScaler(),RandomForestRegressor(n_estimators=200,max_depth=20,min_samples_split=2,
                                                       min_samples_leaf=1,min_weight_fraction_leaf=0.001,
                                                       max_features='auto',n_jobs=-1,max_leaf_nodes=256))
# Random Forest not good enough, rmse-mean 110w

Gboost=make_pipeline(RobustScaler(),HistGradientBoostingRegressor(loss='squared_error',learning_rate=0.05,max_iter=200,
                                                                  max_leaf_nodes=256,max_depth=10,min_samples_leaf=1,
                                                                  l2_regularization=2,max_bins=255))

Adaboost=make_pipeline(RobustScaler(),AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=10,min_samples_split=2,
                                                        min_samples_leaf=1),n_estimators=200,learning_rate=0.1,loss='linear'))


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    # Just the basic average model that computes the mean of different models' output

    def __init__(self, models=None):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        # rf.fit(X,y)
        Gboost.fit(X,y)
        Adaboost.fit(X,y)
        xgb_train = xgb.DMatrix(X, label=y)
        xgboost=xgb.train(xgboost_params,xgb_train,num_boost_round=100,evals=[(xgb_train,'train')],verbose_eval=20)
        lgb_train = lgb.Dataset(X, label=y)
        lightGBM=lgb.train(lightGBM_params,lgb_train,num_boost_round=100,verbose_eval=0)
        self.models=[Gboost,Adaboost,xgboost,lightGBM]
        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        output=[]
        predictions=np.zeros(len(X))
        for i in range(len(self.models)):
            if i==2: temp=self.models[i].predict(xgb.DMatrix(X))
            else: temp=self.models[i].predict(X)
            output.append(temp)
        weights=[0.3, 0.1, 0.1, 0.5]
        for i in range(4):
            predictions+=weights[i]*output[i]
        return predictions

class StackingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    # Stacking model, using output of different model as new input and train a meta-model with these new input and
    # the target

    def __init__(self, base_models=None, meta_model=None, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                if model=='xgb':
                    xgb_train=xgb.DMatrix(X[train_index],label=y[train_index])
                    instance=xgb.train(xgboost_params,xgb_train,num_boost_round=100,evals=[(xgb_train,'train')],verbose_eval=20)
                    self.base_models_[i].append(instance)
                    y_pred=instance.predict(xgb.DMatrix(X[holdout_index]))
                elif model=='lgb':
                    lgb_train=lgb.Dataset(X[train_index],label=y[train_index])
                    instance=lgb.train(lightGBM_params,lgb_train,num_boost_round=100,verbose_eval=0)
                    self.base_models_[i].append(instance)
                    y_pred=instance.predict(X[holdout_index])
                else:
                    instance = clone(model)
                    self.base_models_[i].append(instance)
                    instance.fit(X[train_index], y[train_index])
                    y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(xgb.DMatrix(X)) if isinstance(model,xgb.core.Booster) else
                             model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

avearge_model=AveragingModels()
stacking_model=StackingModels(base_models=(Gboost,Adaboost,'xgb','lgb'),meta_model=lasso)

def make_submission(model):
    model.fit(train_data.values,train_label.values)
    x_pred=model.predict(train_data.values)
    train_loss=np.sqrt(mean_squared_error(y_true=train_label,y_pred=x_pred))
    print("Training loss: %f" %train_loss)
    y_pred=model.predict(test_data.values)
    result_df=pd.DataFrame()
    result_df['Id']=pd.Series(range(len(y_pred)))
    result_df['Predicted']=pd.Series(y_pred)
    result_df.to_csv('submission.csv',index=False)

if __name__=='__main__':
    # change model then you can get the training loss and CV loss of any model including basic ones (lasso, ENet, Random
    # Forest, Adaboost and GBDT) and the stacking model
    model=rf
    train_data, train_label, test_data=normal()
    rmse_mean= rmse_cv(model=model, train=train_data, y_train=train_label.values, n_folds=5)
    print("rmse-mean: %f, rmse-std: %f" %(np.mean(rmse_mean), np.std(rmse_mean)))
    make_submission(model)