import sys
from collections import Counter
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

house_df = pd.read_csv(str(ROOT) + "/Data/test.csv")
# columns with misssing values: model, bedrooms, bathrooms, tenure, built_year, no_of_units


def process_bedroom_bathroom_test():
    # 1. For records lacking either 'bathrooms' or 'bedrooms',we used the average number of bedrooms of houses with the same number of bathrooms to fill in the missing slots.
    # And visa versa for the missing bathrooms
    # 2. For records lacking both bathrooms' and 'bedrooms', we used the the average of bedrooms/bathrooms of houses that have 90% - 110% 'area_size'.
    # But before then, we need to process 'bedroom' data with issues of e.g. value of "3+1" string. We convert them by
    # calulating the addtion opeation in the strings

    def process_bedroom(x):
        if isinstance(x, str):
            temp = x.split("+")
            if "" in temp:
                temp.remove("")
            return np.sum([float(i) for i in temp])
        else:
            return x

    house_df["bedrooms"] = house_df["bedrooms"].map(process_bedroom)

    # put apart records missing both bedrooms and bathrooms for latter processing

    bedroom_avail_df = house_df.dropna(subset=["bedrooms"])
    overall_bedroom_ave = np.mean(bedroom_avail_df["bedrooms"])
    bathroom_avail_df = house_df.dropna(subset=["bathrooms"])
    overall_bathroom_ave = np.mean(bathroom_avail_df["bathrooms"])

    for k, v in house_df[
        house_df["bedrooms"].isna() & (~house_df["bathrooms"].isna())
    ].groupby(
        ["bathrooms"]
    ):  # fill bedrooms
        temp = bedroom_avail_df[bedroom_avail_df["bathrooms"] == k]["bedrooms"]
        if len(temp) == 0:
            house_df.loc[v.index, "bedrooms"] = overall_bedroom_ave
            # special case: cannot find house record with same number of bathrooms, use the overall average
        else:
            temp_counter = Counter(list(temp))
            calculus = 0
            for k0 in temp_counter.keys():
                calculus += int(k0) * temp_counter[k0]
            house_df.loc[v.index, "bedrooms"] = round(
                calculus / len(temp)
            )  # Use average

    for k, v in house_df[
        house_df["bathrooms"].isna() & (~house_df["bedrooms"].isna())
    ].groupby(
        "bedrooms"
    ):  # fill bathrooms
        temp = bathroom_avail_df[bathroom_avail_df["bedrooms"] == k]["bathrooms"]
        if len(temp) == 0:
            house_df.loc[v.index, "bathrooms"] = overall_bathroom_ave
            # special case: cannot find house record with same number of bedrooms, use the overall average
        else:
            temp_counter = Counter(list(temp))
            calculus = 0
            for k0 in temp_counter.keys():
                calculus += int(k0) * temp_counter[k0]
            house_df.loc[v.index, "bathrooms"] = round(
                calculus / len(temp)
            )  # Use average

    area_avail_df = house_df.dropna(subset=["area_size"])
    # actually no data lacks of 'area_size', we just keep the above format uniform
    for idx, row in house_df[
        house_df["bedrooms"].isna() & house_df["bathrooms"].isna()
    ].iterrows():
        area_range = [0.9 * row["area_size"], 1.1 * row["area_size"]]
        temp = area_avail_df[
            (area_avail_df["area_size"] >= area_range[0])
            & (area_avail_df["area_size"] <= area_range[1])
        ]
        if len(temp) == 0:
            house_df.loc[idx, "bedrooms"] = np.mean(area_avail_df["bedrooms"])
            house_df.loc[idx, "bathrooms"] = np.mean(area_avail_df["bathrooms"])
            # special case: cannot find house record with area_size in this range, use the overall average
        else:
            house_df.loc[idx, "bedrooms"] = round(np.mean(temp["bedrooms"]))
            house_df.loc[idx, "bathrooms"] = round(np.mean(temp["bathrooms"]))

    # print(len(house_df) - house_df['bedrooms'].count())
    # print(len(house_df) - house_df['bathrooms'].count())
    print("After processing bedroom and bathroom:", len(house_df))


def process_no_of_units_test():
    # dealing with missing data of no_of_units, we use the record with the same 'name' to fill missing slots
    # Otherwise we fill them with the average of units of different type, respectively. (No missing value in 'type')
    no_of_units_avail_df = house_df.dropna(subset=["no_of_units"])
    ave_units_apt = np.mean(
        no_of_units_avail_df[no_of_units_avail_df["type"] == "apartment"]["no_of_units"]
    )
    ave_units_condo = np.mean(
        no_of_units_avail_df[no_of_units_avail_df["type"] == "condominium"][
            "no_of_units"
        ]
    )

    for k, v in house_df[house_df["no_of_units"].isna()].groupby(["name"]):
        temp = no_of_units_avail_df[no_of_units_avail_df["name"] == k]["no_of_units"]
        if len(temp) == 0:
            continue
            # no name matched, skip them now
        else:
            house_df.loc[v.index, "no_of_units"] = np.mean(temp)
            # Note that every record with same name should have same number of no_of_units

    for k, v in house_df[house_df["no_of_units"].isna()].groupby(["type"]):
        if k == "apartment":
            house_df.loc[v.index, "no_of_units"] = ave_units_apt
        else:
            house_df.loc[v.index, "no_of_units"] = ave_units_condo

    print("After processing no_of_units:", len(house_df))


def process_tenure_test():
    # all missing values in 'tenuer' are replaced with number 99 (not str)
    # We want to calculate how many years we can use for this house from now on
    # and divide these years into three categories: <500 years, 500~5000 years and >5000 years
    # We map them to 0, 1, 2
    house_df.fillna({"tenure:99"})

    def catogory(tenure_str: str) -> int:
        def tenure_handle(tenure: str) -> int:
            # Calculate how many years we could use
            # Tenure only contains four formats: "freehold" "leasehold/XX years" "XX years" and "XX years from XX"
            string_list = tenure.split(" ")
            if string_list[0] == "freehold":
                return 9999  # "freehold"
            elif string_list[0][0].isalpha():
                return int(string_list[0][10:])  # "leasehold/XX years"
            elif string_list[0][0].isdigit() and len(string_list) == 2:
                return int(string_list[0])  # "XX years"
            else:
                return int(string_list[0]) - (
                    2022 - int(string_list[-1][-4:])
                )  # "XX years from XX"

        temp = tenure_handle(tenure_str)
        if temp < 500:
            return 0
        elif temp < 5000:
            return 1
        else:
            return 2

    house_df["tenure"] = house_df["tenure"].map(lambda x: catogory(x))
    house_df = pd.get_dummies(house_df, columns=["tenure"], prefix="tenure")


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
        lambda x: 0  #'ccr' situation
        if x["district"] in [9, 10, 11]
        or x["planning_area"] in ["downtown core", "southern islands"]
        else 1  # 'rcr' situation
        if x["planning_area"] in central_region
        else 2,  # 'ocr' situation
        axis=1,
    )
    house_df["ccr"] = pd.Series(
        [1 if x == 0 else 0 for x in house_df["market_segment"]], index=house_df.index
    )
    house_df["rcr"] = pd.Series(
        [1 if x == 1 else 0 for x in house_df["market_segment"]], index=house_df.index
    )
    house_df["ocr"] = pd.Series(
        [1 if x == 2 else 0 for x in house_df["market_segment"]], index=house_df.index
    )
    house_df.drop(columns=["market_segment"], inplace=True)


def add_nearest_distance_test():
    # add nearest distance (meter) of a house to a certain type of place, 6 features in total
    # The distance between every degree of latitude is 111km while that of longitude varies with different latitutes.
    # The maximum distance between every degree of longitude is at the equator, also 111km
    # And Singapore happens to be an island near the equator (the latitude and longitude span between locations is very
    # small), the error of using the data at the equator is controllable

    commercial_center_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-commerical-centres.csv"
    )
    hawker_center_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-gov-markets-hawker-centres.csv"
    )
    primary_school_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-primary-schools.csv"
    )
    secondary_school_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-secondary-schools.csv"
    )
    shopping_mall_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-shopping-malls.csv"
    )
    train_station_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-train-stations.csv"
    )
    col_name = [
        "commercial_center",
        "hawker_center",
        "primary_school",
        "secondary_school",
        "shopping_mall",
        "train_station",
    ]
    auxiliary_df = [
        commercial_center_df,
        hawker_center_df,
        primary_school_df,
        secondary_school_df,
        shopping_mall_df,
        train_station_df,
    ]
    for i in range(6):
        print(col_name[i])
        distance = []
        for j in range(len(house_df)):
            if j % 1000 == 0:
                print("processed: %d/%d" % (j, len(house_df)))
            # need to use iloc because it represents absolute position, however loc refers to the original index
            # that we already messed up
            min_dis = np.min(
                [
                    np.sqrt(
                        (
                            (house_df["lat"].iloc[j] - auxiliary_df[i]["lat"].iloc[k])
                            * 111000
                        )
                        ** 2
                        + (
                            (house_df["lng"].iloc[j] - auxiliary_df[i]["lng"].iloc[k])
                            * 111000
                        )
                        ** 2
                    )
                    for k in range(len(auxiliary_df[i]))
                ]
            )
            distance.append(min_dis)
        house_df[col_name[i]] = pd.Series(distance, index=house_df.index)
        # note this index thing, need to match the original index in the dataframe
        print(house_df[col_name[i]])


def add_no_of_POI_test():
    commercial_center_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-commerical-centres.csv"
    )
    hawker_center_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-gov-markets-hawker-centres.csv"
    )
    primary_school_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-primary-schools.csv"
    )
    secondary_school_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-secondary-schools.csv"
    )
    shopping_mall_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-shopping-malls.csv"
    )
    train_station_df = pd.read_csv(
        str(ROOT) + "/Data/auxiliary-data/sg-train-stations.csv"
    )
    auxiliary_df = [
        commercial_center_df,
        hawker_center_df,
        primary_school_df,
        secondary_school_df,
        shopping_mall_df,
        train_station_df,
    ]
    poi_df = pd.concat(auxiliary_df, ignore_index=True)
    no_of_poi_5km = []
    for i in range(len(house_df)):
        if i % 100 == 0:
            print("processed: %d" % (i))
        dis = []
        for j in range(len(poi_df)):

            dis.append(
                np.sqrt(
                    ((house_df["lat"].iloc[i] - poi_df["lat"].iloc[j]) * 111000) ** 2
                    + ((house_df["lng"].iloc[i] - poi_df["lng"].iloc[j]) * 111000) ** 2
                )
            )

        no_of_poi_5km.append(len([x for x in dis if x <= 1000]))

    house_df["no_of_poi_5km"] = pd.Series(no_of_poi_5km)


def remove_features_test():
    # Remove 'listing_id' 'market_segment' 'type_of_areas' 'eco_category' 'accessibility' 'date_listed' 'built_year'
    # 'name' 'street' 'model' 'lat' 'lng'
    house_df.drop(
        columns=[
            "listing_id",
            "type_of_area",
            "eco_category",
            "accessibility",
            "lat",
            "lng",
            "model",
            "date_listed",
            "built_year",
            "name",
            "street",
        ],
        inplace=True,
    )


process_bedroom_bathroom_test()
process_no_of_units_test()
process_bedroom_bathroom_test()
process_market_segment()
add_nearest_distance_test()
add_no_of_POI_test()
remove_features_test()

house_df.to_csv("processed_test.csv", index=False)

