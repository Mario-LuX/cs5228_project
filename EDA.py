import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import *
import seaborn as sns

house_df=pd.read_csv('Data/train.csv')
units_df = house_df.dropna(subset=['no_of_units'])
bathroom_avail_df = house_df.dropna(subset=['bathrooms'])
# This data should be pre-defined to avoid the effect of newly-filled data
# And we want to avoid the effects that processing a certain feature could bring to other features, because the deleted
# entries may contain useful information for other features

# Step 1 Handling missing data

def process_bedroom_and_bathroom():
    # There are 73 records missing both 'bathrooms' and 'bedrooms', drop them
    # We used the average number of bedrooms for houses with the same number of bathrooms to fill in the missing slots.
    # And visa versa for the missing bathrooms
    # But before then, we need to process 'bedroom' data with issues of e.g. value of "3+1" string. We convert them by
    # calulating the addtion opeation in the strings

    house_df['bedrooms'] = house_df['bedrooms'].map(lambda x: np.sum([float(i) for i in x.split("+")])
                                                    if isinstance(x, str) else x)
    house_df.drop(index=house_df[house_df['bedrooms'].isna()][house_df['bathrooms'].isna()].index, inplace=True)
    # delete records missing both bedrooms and bathrooms
    bedroom_avail_df = house_df.dropna(subset=['bedrooms'])

    for k, v in house_df[house_df['bedrooms'].isna()].groupby(['bathrooms']):  # fill bedrooms
        temp = bedroom_avail_df[bedroom_avail_df['bathrooms'] == k]['bedrooms']
        if (len(temp) == 0):
            house_df.drop(index=v.index, inplace=True)
            # special case: cannot find house record with same number of bathrooms, drop these records
        else:
            # house_df.loc[v.index,'bedrooms']=list(temp.mode())[0] # Use mode
            temp_counter = Counter(list(temp))
            calculus = 0
            for k0 in temp_counter.keys():
                calculus += (int(k0) * temp_counter[k0])
            house_df.loc[v.index, 'bedrooms'] = round(calculus / len(temp)) # Use average

    for k, v in house_df[house_df['bathrooms'].isna()].groupby('bedrooms'):  # fill bathrooms
        temp = bathroom_avail_df[bathroom_avail_df['bedrooms'] == k]['bathrooms']
        if (len(temp) == 0):
            house_df.drop(index=v.index, inplace=True)
            # special case: cannot find house record with same number of bedrooms, drop these records
        else:
            # house_df.loc[v.index,'bathrooms']=list(temp.mode())[0] # Use mode
            temp_counter = Counter(list(temp))
            calculus = 0
            for k0 in temp_counter.keys():
                calculus += (int(k0) * temp_counter[k0])
            house_df.loc[v.index, 'bathrooms'] = calculus / len(temp)  # Use average
    # print(len(house_df) - house_df['bedrooms'].count())
    # print(len(house_df) - house_df['bathrooms'].count())
    print("After processing bedroom and bathroom", len(house_df))

def process_no_of_units():
    # dealing with missing data of no_of_units, we use the record with the same 'name' to fill missing slots
    # Otherwise we drop these records
    count=0
    for k,v in house_df[house_df['no_of_units'].isna()].groupby(['name']):
        temp=units_df[units_df['name']==k]['no_of_units']
        if(len(temp)==0):
            # print(k, len(v))
            house_df.drop(index=v.index,inplace=True) #no name matched, drop them (a large part)
        else:
            count+=len(v)
            house_df.loc[v.index,'no_of_units']=np.mean(temp)
            #Note that every record with same name should have same number of no_of_units
    # print(len(house_df)-house_df['no_of_units'].count())
    print("We save %d record" %(count))
    print("After processing no_of_units", len(house_df))

def drop_missing_slots():
    # Here we decide to drop missing slots of 'tenure' and 'area_size'
    house_df.dropna(subset=['tenure','area_size'],inplace=True)
    print("After processing tenure etc", len(house_df))

# Step 2 Handling problems with the feature, process features

def process_tenure():
    # We want to calculate how many years we can use for this house from now on
    # and divide these years into three categories: <500 years, 500~5000 years and >5000 years
    # And apply one-hot encoding to these categories with column name of ['tenure_0', 'tenure_1', 'tenure_2']
    def tenure_handle(tenure: str) -> int:
        # Calculate how many years we could use
        # Tenure only contains four formats: "freehold" "leasehold/XX years" "XX years" and "XX years from XX"
        string_list = tenure.split(" ")
        if string_list[0] == 'freehold':
            return 9999  # "freehold"
        elif string_list[0][0].isalpha():
            return int(string_list[0][10:])  # "leasehold/XX years"
        elif string_list[0][0].isdigit() and len(string_list) == 2:
            return int(string_list[0])  # "XX years"
        else:
            return int(string_list[0]) - (2022 - int(string_list[-1][-4:]))  # "XX years from XX"

    def catogory(temp: int) -> int:
        if temp<500: return 0
        elif temp<5000: return 1
        else: return 2
    house_df['tenure'] = house_df['tenure'].map(lambda x: tenure_handle(x))
    house_df['tenure'] =house_df['tenure'].map(lambda x: catogory(x))
    one_hot=pd.get_dummies(house_df['tenure'],prefix='tenure')
    plot_list=[]
    for i in one_hot.columns:
        house_df[i]=pd.Series(list(one_hot[i]),index=house_df.index)
        plot_list.append(np.sum(one_hot[i]))

    def func(pct, allvals):
        absolute = int(pct / 100.0 * np.sum(allvals))
        return "{:.1f}% ( {:d} )".format(pct, absolute)

    wedges, texts, autotexts = plt.pie(
        plot_list,
        labels=['short-term','long-term','freehold'],
        autopct=lambda pct: func(pct, plot_list),
        colors=plt.cm.Pastel1.colors,
        startangle=140,
        textprops={"fontsize": 12, "color": "k"},
    )

    plt.title("Distribution of How Many Years We Can Use")
    plt.show()
    house_df.drop(columns=['tenure'],inplace=True)

def process_type():
    # binarize 'type' feature
    house_df['type']=house_df['type'].map(lambda x: 0 if x=='apartment' else 1)

def process_market_segment():
    # we divide market_segment feature into 'ocr' 'rcr' 'ccr' three parts
    # According to these links below:
    # https://www.ura.gov.sg/-/media/Corporate/Property/REALIS/realis-maps/map_ccr.pdf
    # https://data.gov.sg/dataset/private-residential-property-transactions-in-rest-of-central-region-quarterly
    # two links for how to divide these three types
    # https://en.wikipedia.org/wiki/Central_Region,_Singapore for how to distinguish central region
    central_region = [
        "Bishan",
        "Bukit Merah",
        "Bukit Timah",
        "Geylang",
        "Kallang",
        "Marine Parade",
        "Novena",
        "Queenstown",
        "Southern Islands",
        "Tanglin",
        "Toa Payoh",
        "Downtown Core",
        "Marina East",
        "Marina South",
        "Museum",
        "Newton",
        "Orchard",
        "Outram",
        "River Valley",
        "Rochor",
        "Singapore River",
        "Straits View",
    ]
    central_region = sorted([i.lower() for i in central_region])
    house_df["market_segment"] = house_df[["district", "planning_area"]].apply(
        lambda x: 0 #'ccr' situation
        if x["district"] in [9, 10, 11]
        or x["planning_area"] in ["downtown core", "southern islands"]
        else 1 # 'rcr' situation
        if x["planning_area"] in central_region
        else 2, # 'ocr' situation
        axis=1,
    )
    house_df['ccr'] = pd.Series([1 if x == 0 else 0 for x in house_df['market_segment']], index=house_df.index)
    house_df['rcr'] = pd.Series([1 if x == 1 else 0 for x in house_df['market_segment']], index=house_df.index)
    house_df['ocr'] = pd.Series([1 if x == 2 else 0 for x in house_df['market_segment']], index=house_df.index)
    house_df.drop(columns=['market_segment'], inplace=True)


# Step 3 Add new features(maybe)

def add_nearest_distance():
    # add nearest distance (meter) of a house to a certain type of place, 6 features in total
    # The distance between every degree of latitude is 111km while that of longitude varies with different latitutes.
    # The maximum distance between every degree of longitude is at the equator, also 111km
    # And Singapore happens to be an island near the equator (the latitude and longitude span between locations is very
    # small), the error of using the data at the equator is controllable

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
        distance=[]
        for j in range(len(house_df)):
            if(j%1000==0): print("processed: %d/%d" %(j,len(house_df)))
            # need to use iloc because it represents absolute position, however loc refers to the original index
            # that we already messed up
            min_dis=np.min([np.sqrt(((house_df['lat'].iloc[j]-auxiliary_df[i]['lat'].iloc[k])*111000)**2+
                            ((house_df['lng'].iloc[j]-auxiliary_df[i]['lng'].iloc[k])*111000)**2)
                     for k in range(len(auxiliary_df[i]))])
            distance.append(min_dis)
        house_df[col_name[i]]=pd.Series(distance,index=house_df.index)
        # note this index thing, need to match the original index in the dataframe
        # print(house_df[col_name[i]])

def remove_features():
    # Remove 'listing_id' 'market_segment' 'type_of_areas' 'eco_category' 'accessibility' 'date_listed' 'built_year'
    # 'name' 'street' 'model' 'lat' 'lng'
    house_df.drop(columns=['listing_id', 'market_segment', 'type_of_area', 'eco_category', 'accessibility', 'lat', 'lng',
                           'model', 'date_listed', 'built_year', 'name', 'street'], inplace=True)


def add_no_of_POI(radius=1):
    # Add number of POI in am area of "radius" kilometers centered with each house
    commercial_center_df = pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-commerical-centres.csv')
    hawker_center_df = pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-gov-markets-hawker-centres.csv')
    primary_school_df = pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-primary-schools.csv')
    secondary_school_df = pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-secondary-schools.csv')
    shopping_mall_df = pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-shopping-malls.csv')
    train_station_df = pd.read_csv('Data/auxiliary-data/auxiliary-data/sg-train-stations.csv')
    auxiliary_df = [commercial_center_df, hawker_center_df, primary_school_df, secondary_school_df, shopping_mall_df,
                    train_station_df]
    poi_df = pd.concat(auxiliary_df, ignore_index=True)
    no_of_poi = []
    for i in range(len(house_df)):
        if (i % 100 == 0):
            print("processed: %d" % (i))
        dis = []
        for j in range(len(poi_df)):
            dis.append(np.sqrt(((house_df['lat'].iloc[i] - poi_df['lat'].iloc[j]) * 111000) ** 2 +
                               ((house_df['lng'].iloc[i] - poi_df['lng'].iloc[j]) * 111000) ** 2))
        no_of_poi.append(len([x for x in dis if x <= radius*1000]))
    house_df['no_of_poi'] = pd.Series(no_of_poi,index=house_df.index)

if __name__=='__main__':
    print(len(house_df))
    process_bedroom_and_bathroom()
    process_no_of_units()
    drop_missing_slots()
    process_tenure()
    process_market_segment()
    process_type()
    add_nearest_distance()
    add_no_of_POI(radius=1)
    remove_features()
    house_df.to_csv('train.csv',index=False)
    sns.heatmap(house_df[['bedrooms', 'bathrooms', 'area_size']].corr())
    plt.show()