import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

house_data=pd.read_csv('processed_train.csv')
test_data=pd.read_csv('processed_test.csv')

train_columns=['bedrooms','bathrooms','no_of_units','area_size',
               'commercial_center','hawker_center','primary_school','secondary_school','shopping_mall','train_station',
               'no_of_poi','tenure_0','tenure_1','tenure_2','ccr','rcr','ocr','type']
train_data=house_data[train_columns]
train_label=house_data['price']
test_data=test_data[train_columns]
all_data=pd.concat((train_data,test_data)).reset_index(drop=True)
print(train_data.shape)
print(test_data.shape)
print(all_data.shape)

def price_analysis(train_label):
    # A simple function to analyze the distribution of label
    # looks like after the log-transform, it fits a gamma distribution
    sns.distplot(train_label , fit=stats.gamma)

    # Get the fitted parameters used by the function
    (alpha, loc, beta) = stats.gamma.fit(train_label)
    print( '\n alpha= {:.2f}  loc: {:.2f} beta= {:.2f}\n'.format(alpha, loc, beta))
    #Now plot the distribution
    plt.legend(['Gamma dist. (alpha= {:.2f}  loc: {:.2f} beta= {:.2f} )'.format(alpha, loc, beta)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')

    plt.show()

    train_label=np.log1p(train_label)
    sns.distplot(train_label , fit=stats.gamma)
    (alpha, loc, beta) = stats.gamma.fit(train_label)
    print( '\n alpha= {:.2f}  loc: {:.2f} beta= {:.2f}\n'.format(alpha, loc, beta))
    #Now plot the distribution
    plt.legend(['Gamma dist. (alpha= {:.2f}  loc: {:.2f} beta= {:.2f} )'.format(alpha, loc, beta)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    plt.show()

def statistical_solution(combine=False):
    # statistical solution, the parameter means whether we use data obtained from crawler
    feature_chosen=["name", "street", "area_size", "bedrooms", "bathrooms","price","no_of_units"]
    train_dataset = pd.read_csv("Data/train.csv")
    crawler_dataset=pd.read_csv('crawler.csv')
    train_dataset=train_dataset[feature_chosen]
    crawler_dataset=crawler_dataset[feature_chosen]
    if combine:
        train_dataset=pd.concat([train_dataset,crawler_dataset],ignore_index=True)
    test_dataset = pd.read_csv("Data/test.csv")

    best_dataset = pd.read_csv("submission_best.csv")

    test_dataset["Predicted"] = test_dataset[
        ["name", "street", "area_size", "bedrooms", "bathrooms","no_of_units"]].apply(
        lambda x: train_dataset[
            (train_dataset["area_size"] == x["area_size"])
            & (train_dataset['name'] == x['name'])
            # & (train_dataset['street']==x['street'])
            # & (train_dataset['no_of_units'] == x['no_of_units'])
            # & (train_dataset['bedrooms']==x['bedrooms'])
            # & (train_dataset['bathrooms']==x['bathrooms'])
            ]["price"].mean(),
        axis=1,
    )
    print(test_dataset['Predicted'].count())
    test_dataset["Predicted"].fillna(best_dataset["Predicted"], inplace=True)
    test_dataset["Predicted"].to_csv("sub_statistical.csv", index=True, index_label="Id")

if __name__=='__main__':
    price_analysis(train_label)
    statistical_solution(combine=False)