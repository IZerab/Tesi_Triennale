import pandas as pd
import numpy as np
# this document is intended to set the values indicated by the italian minister of health
# the levels are reported below, note that CO is definitely higher that the others
# Quick Ref: buono = good, Discreto = Discrete, Moderato = moderate, Scadente = poor, Pessimo = bad

# I define a support dictionary
subClas = {}
subClas["CO"] = [
    {"low": 0, "high": 5000, "name": "Buono"},
    {"low": 5001, "high": 7500, "name": "Discreto"},
    {"low": 7501, "high": 10000, "name": "Moderato"},
    {"low": 10001, "high": 20000, "name": "Scadente"},
    {"low": 20001, "high": float("inf"),  "name": "Pessimo"}
    ]

subClas["NO2"] = [
    {"low": 0, "high": 40, "name": "Buono"},
    {"low": 41, "high": 100, "name": "Discreto"},
    {"low": 101, "high": 200, "name": "Moderato"},
    {"low": 201, "high": 400, "name": "Scadente"},
    {"low": 401, "high": float("inf"), "name": "Pessimo"}
    ]

subClas["PM10"] = [
    {"low": 0, "high": 20, "name": "Buono"},
    {"low": 21, "high": 35, "name": "Discreto"},
    {"low": 36, "high": 50, "name": "Moderato"},
    {"low": 51, "high": 100, "name": "Scadente"},
    {"low": 101, "high": float("inf"), "name": "Pessimo"}
    ]
# they are the same!!
subClas["PM2_5"] = [
    {"low": 0, "high": 20, "name": "Buono"},
    {"low": 21, "high": 35, "name": "Discreto"},
    {"low": 36, "high": 50, "name": "Moderato"},
    {"low": 51, "high": 100, "name": "Scadente"},
    {"low": 101, "high": float("inf"), "name": "Pessimo"}
    ]

subClas["O3"] = [
    {"low": 0, "high": 80, "name": "Buono"},
    {"low": 81, "high": 120, "name": "Discreto"},
    {"low": 121, "high": 180, "name": "Moderato"},
    {"low": 181, "high": 240, "name": "Scadente"},
    {"low": 241, "high": float("inf"), "name": "Pessimo"}
    ]

subClas["SO2"] = [
    {"low": 0, "high": 100, "name": "Buono"},
    {"low": 101, "high": 200, "name": "Discreto"},
    {"low": 201, "high": 350, "name": "Moderato"},
    {"low": 351, "high": 500, "name": "Scadente"},
    {"low": 501, "high": float("inf"), "name": "Pessimo"}
    ]


def air_cataloguer(df, pollutants):
    """
    This function assign a label to the raw data following the indication of the italian minister of health
    The pollutants are inserted in the string pollutants
    The result of this programme is a df!!
    """
    dict = {}  # support dict meant not to overwrite subCLas

    for pol in pollutants:
        dict[pol] = pd.DataFrame(subClas[pol])  # data frame di supporto per suddividere il classificatore
        bins = list(dict[pol].loc[:, "high"])
        bins.insert(0, 0)
        df.loc[:, "Livello Inquinanti"] = pd.cut(df.loc[:, "Valore"], bins, labels=dict[pol].loc[:, 'name'])
    return df
