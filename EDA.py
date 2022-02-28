import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import *
import seaborn as sns


house_df=pd.read_csv('Data/train.csv')
print(len(house_df))
# Step 1 Handling missing data
# drop missing data of 'bedrooms' 'bathrooms' 'tenure' 'no_of_units' 'area_size' 'model'
temp_df=house_df.dropna(subset=['bedrooms','bathrooms','tenure','no_of_units','area_size','model'])
print(len(temp_df))
# bedrooms bathrooms 都缺失的只有73条数据，删除
# bedrooms和 bathrooms 可以互补，用bathroom不缺的房屋加权平均bedroom数量去补充 反之亦然
# no_of_units可以补充，可以使用同一个name下的数据来填充, 发现大部分是没有相同名字的有效记录但是少部分有可以填充

house_df['bedrooms']=house_df['bedrooms'].map(lambda x: np.sum([float(i) for i in x.split("+")])
                                                if isinstance(x,str) else x)
#要实现补全功能要先处理bedrooms里面的不符合要求值
bedroom_avail_df=house_df.dropna(subset=['bedrooms']) #不能掺杂补全后的数据造成影响
bathroom_avail_df=house_df.dropna(subset=['bathrooms'])
units_df=house_df.dropna(subset=['no_of_units'])
house_df.drop(index=house_df[house_df['bedrooms'].isna()][house_df['bathrooms'].isna()].index,inplace=True)
#两者都缺少的记录丢弃

for k,v in house_df[house_df['bedrooms'].isna()].groupby(['bathrooms']): #补全bedrooms缺失
    temp=bedroom_avail_df[bedroom_avail_df['bathrooms']==k]['bedrooms']
    if(len(temp)==0): house_df.drop(index=v.index, inplace=True)  # 有可能在存在记录中没有对应相同的记录，则删除
    else:
        # house_df.loc[v.index,'bedrooms']=list(temp.mode())[0] # 用众数代替
        # 在更新过程中注意数据类型，要为纯数据类型，很多时候处理出来是pandas.Series
        temp_counter=Counter(list(temp))
        calculus=0
        for k0 in temp_counter.keys():
            calculus+=(int(k0)*temp_counter[k0])
        house_df.loc[v.index,'bedrooms']=calculus/len(temp)

for k,v in house_df[house_df['bathrooms'].isna()].groupby('bedrooms'): #补全bathrooms缺失
    temp = bathroom_avail_df[bathroom_avail_df['bedrooms'] == k]['bathrooms']
    if (len(temp) == 0): house_df.drop(index=v.index, inplace=True)  # 有可能在存在记录中没有对应相同的记录，则删除
    else:
        # house_df.loc[v.index,'bathrooms']=list(temp.mode())[0] #用众数代替
        temp_counter=Counter(list(temp))
        calculus = 0
        for k0 in temp_counter.keys():
            calculus += (int(k0) * temp_counter[k0])
        house_df.loc[v.index, 'bathrooms'] = calculus / len(temp) #用加权平均

print(len(house_df)-house_df['bedrooms'].count())
print(len(house_df)-house_df['bathrooms'].count())
print("After processing bedroom and bathroom", len(house_df))

#处理no_of_units, 因为name没有缺失值不需担心两者都缺失的情况
#这里只能挽回27条数据
for k,v in house_df[house_df['no_of_units'].isna()].groupby(['name']):
    temp=units_df[units_df['name']==k]['no_of_units']
    if(len(temp)==0):
        # print(k, len(v))
        house_df.drop(index=v.index,inplace=True) #没有对应记录删除（大多数都没有）
    else:
        house_df.loc[v.index,'no_of_units']=np.mean(temp) #数据中所有同name记录中no_of_units都相同
print(len(house_df)-house_df['no_of_units'].count())
print("After processing no_of_units", len(house_df))

house_df.dropna(subset=['tenure','area_size','model'],inplace=True)
print("After processing tenure etc", len(house_df))

'''
x_label=[k for (k,_) in house_df[house_df['bathrooms']==1.0].groupby(['bedrooms'])]
heights=[len(v) for (_,v) in house_df[house_df['bathrooms']==1.0].groupby(['bedrooms'])]
plt.bar(x_label,heights)
plt.title(str(len(bedroom_nan_df[bedroom_nan_df['bathrooms']==1.0])))
plt.show()

for k,v in units_nan_df.groupby(['name']):
    print(k,len(v),"is available:", house_df[house_df['name']==k]['no_of_units'].count()>0)

house_df.dropna(subset=['bedrooms','bathrooms','tenure','no_of_units','area_size','model'],inplace=True)
print(len(house_df))
'''


# Step 2 Handling problems with the feature and outliers
# Remove 'listing_id' 'market_segment' 'type_of_areas' 'eco_category' 'accessibility' 'date_listed'
house_df.drop(columns=['listing_id','market_segment','type_of_area','eco_category','accessibility','date_listed'],inplace=True)

# Remove 'built_year' because large ratio of missing values
# maybe remove 'street' and 'name'
# region 一级粗划分 district（27）和planning_area（38）是两种不同的细一些的划分, subzone是根据planning_area的更细化分
'''
print(len(set(house_df['district'])))
print(len(set(house_df['planning_area'])))

for col in house_df.columns:
    # print(col,len(house_df)-house_df[col].count())
    if col!="listing_id": print(col, set(house_df[col].dropna()))
plt.hist(house_df['price'],bins=100)
plt.show()
'''

# Step 3 Handling categorical feature, process features and add new features(maybe)

# Compute the actual usable period of houses (from 2022, measurement: years)
def tenure_handle(tenure: str) -> int:
    # Tenure里面仅包含四种格式 "freehold" "leasehold/XX years" "XX years" "XX years from XX"
    string_list=tenure.split(" ")
    if string_list[0]=='freehold': return 9999 # "freehold"
    elif string_list[0][0].isalpha(): return int(string_list[0][10:]) #"leasehold/XX years"
    elif string_list[0][0].isdigit() and len(string_list)==2: return int(string_list[0]) # "XX years"
    else: return int(string_list[0])-(2022-int(string_list[-1][-4:])) #"XX years from XX"


house_df['tenure']=house_df['tenure'].map(lambda x: tenure_handle(x))
# print(set(house_df['tenure']))

# 每个纬度之间相差一度相差111km，每个经度之间每隔一度随着纬度不同不一样在赤道处最大也是111km
# 而新加坡正好是处于赤道附近的一个小岛（地点之间经纬度跨度很笑），使用赤道处数据误差可控
'''
commercial_center_df=pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-commerical-centres.csv')
hawker_center_df=pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-gov-markets-hawker-centres.csv')
primary_school_df=pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-primary-schools.csv')
secondary_school_df=pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-secondary-schools.csv')
shopping_mall_df=pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-shopping-malls.csv')
train_station_df=pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-train-stations.csv')
col_name=['commercial_center','hawker_center','primary_school','secondary_school','shopping_mall','train_station']
auxiliary_df=[commercial_center_df,hawker_center_df,primary_school_df,secondary_school_df,shopping_mall_df,train_station_df]
for i in range(6):
    print(col_name[i])
    n=len(auxiliary_df[i])
    distance=[]
    for j in range(len(house_df)):
        if(j%1000==0): print("processed: %d/%d" %(j,len(house_df)))
        # [j]表示使用series的index，但是之前处理house_df已经乱了
        min_dis=np.min([np.sqrt(((house_df['lat'].iloc[j]-auxiliary_df[i]['lat'].iloc[k])*111000)**2+
                        ((house_df['lng'].iloc[j]-auxiliary_df[i]['lng'].iloc[k])*111000)**2)
                 for k in range(len(auxiliary_df[i]))])
        distance.append(min_dis)
    house_df[col_name[i]]=pd.Series(distance,index=house_df.index) #注意dataframe和series的index和纯粹位置下标的区别
    print(house_df[col_name[i]])
house_df.to_csv('train_new.csv',index=False)
'''
# Add new features: compute neareast distance(meters) to different landmarks (6 features)

#Categorical data: 'type' 'model' 'street' 'region' 'planning_area' 'subzone' (to be decided)

# plt.hist(house_df['no_of_units'],bins=20)
sns.heatmap(house_df[['bedrooms','bathrooms','area_size']].corr())
plt.show()
