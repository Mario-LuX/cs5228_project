import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange
from scipy import stats
from scipy.special import boxcox1p
from torch.autograd import Variable
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

def normal():
    # Get the data to feed the model
    # Just use the data after preprocessing (i.e. use 'market_segment' division and abandon 'district' etc)
    train_data=pd.read_csv('processed_train.csv')
    test_data=pd.read_csv('processed_test.csv')

    train_label=train_data['price']
    test_data.drop(columns=['district','planning_area','subszone','region'],inplace=True)
    train_data.drop(columns=['district','planning_area','subszone','region','price'],inplace=True)
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

def box_cox(train_data,test_data):
    num_train = len(train_data)
    # Apply box-cox transformation to every feature
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

# train_data, train_label, test_data=add_crawler()
train_data, train_label, test_data=district()
# train_data, test_data=process_coordinate(train_data,test_data)
# train_data, test_data=box_cox(train_data,test_data)

X_train = train_data.to_numpy()
y_train = train_label.to_numpy()
X_test = test_data.to_numpy()
len_input=X_train.shape[1]

X_train = torch.tensor(X_train).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
X_test = torch.tensor(X_test).float().to(device)

class network(nn.Module):
    # Nerual Network model with skip connection
    def __init__(self):
        super(network,self).__init__()
        self.layer1=nn.Linear(len_input,50)
        self.layer2=nn.Linear(50,128)
        self.layer3=nn.Linear(128,50)
        self.l1=nn.Linear(128,128)
        self.layer4=nn.Linear(50,1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.Tanh=nn.Tanh()
        self.dropout1=nn.Dropout(p=0.1)
        self.dropout2=nn.Dropout(p=0.1)

    def forward(self,x):
        x=self.layer1(x)
        temp=torch.clone(x)
        # x=self.dropout1(x)
        x=self.relu(x)
        x=self.layer2(x)
        # x = self.dropout2(x)
        x=self.relu(x)
        x=self.layer3(x)
        x+=temp
        x=self.relu(x)
        x=self.layer4(x)
        return x.squeeze()

class RMSELoss(nn.Module):
    # There is no RMSE loss in torch, only MSE loss
    # We need to override the class
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.layer1=nn.Linear(len_input,50)
        self.layer2=nn.Linear(50,50)
        self.layer3=nn.Linear(50,1)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        x=self.relu(x)
        x=self.layer3(x)
        return x.squeeze()

model=network().to(device)
# model = net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.01)
# Adam optimizer doesn't need to specify momentum
loss_func = RMSELoss()

def cross_validation(X,y):
    # Doing cross validation and visualization of our network
    kfold=KFold(n_splits=5,shuffle=True)
    train_loss=[[] for i in range(5)]
    cv_loss=[[] for i in range(5)]
    fold=0
    for train_index, test_index in kfold.split(X):
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        for i in trange(10000):
            X_train = Variable(X_train)
            y_train = Variable(y_train)
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = loss_func(y_pred, y_train)
            train_loss[fold].append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                output = model(X_test)
                fold_loss = loss_func(output, y_test).cpu().detach().numpy()
                cv_loss[fold].append(fold_loss)
            # if (i % 100 == 0): print(loss.cpu().detach().numpy())
        fold+=1
    train_loss=np.mean(train_loss,axis=0)
    cv_loss=np.mean(cv_loss,axis=0)
    print("The Training loss: %f" % train_loss[-1])
    print("The cross validation loss: %f" % cv_loss[-1])
    plt.plot(train_loss,color='b')
    plt.plot(cv_loss,color='r')
    plt.title("Training loss and CV loss through epochs")
    plt.ylabel("Loss(RMSE)")
    plt.xlabel("Epochs")
    plt.legend(["Training loss","CV loss"])
    plt.show()

def training_and_submit(X_train,y_train,X_test):
    # Train the model and make submission
    for i in trange(10000):
        X_train = Variable(X_train)
        y_train = Variable(y_train)
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_func(y_pred, y_train)
        loss.backward()
        optimizer.step()
        # if(i% 100==0): print(loss.cpu().detach().numpy())

    y_pred = model(X_test)
    y_pred = pd.DataFrame(y_pred.cpu().detach().numpy(), columns=["Predicted"])
    y_pred.to_csv("submission.csv", index=True, index_label="Id")

if __name__=='__main__':
    cross_validation(X_train,y_train)
    training_and_submit(X_train,y_train,X_test)

