from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd

df = pd.read_csv("train.csv").drop(["id", "Row#"], axis=1)

class FeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data = data.copy() 
        data["mean_bee"] = (data["honeybee"] + data["bumbles"] + data["andrena"] + data["osmia"]) / 4
        data["fruitmass_to_seeds"] = data["fruitmass"] / data["seeds"]
        data["fruitmass_to_p"] = data["fruitmass_to_seeds"] ** data["fruitmass"]
        data["round_fruit_set"] = data["fruitset"] / data["seeds"]
        data["fruitset * mean_bee"] = data["mean_bee"] * data["fruitset"]
        data["fruitset / fruitmass"] = data["fruitset"] / data["fruitmass"]
        return data
    