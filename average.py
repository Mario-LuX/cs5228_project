import sys
from pathlib import Path

import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

train_dataset = pd.read_csv(str(ROOT) + "/Data/train.csv")
test_dataset = pd.read_csv(str(ROOT) + "/Data/test.csv")

best_dataset = pd.read_csv(str(ROOT) + "/Data/best_submission.csv")


test_dataset["Predicted"] = test_dataset[
    ["name", "area_size", "bedrooms", "bathrooms"]
].apply(
    lambda x: train_dataset[
        (train_dataset["name"] == x["name"])
        & (train_dataset["area_size"] == x["area_size"])
        & (train_dataset["bedrooms"] == x["bedrooms"])
        & (train_dataset["bathrooms"] == x["bathrooms"])
    ]["price"].mean(),
    axis=1,
)

test_dataset["Predicted"].fillna(best_dataset["Predicted"], inplace=True)

test_dataset["Predicted"].to_csv(
    str(ROOT) + "/Data/submission_average.csv", index=True, index_label="Id"
)

