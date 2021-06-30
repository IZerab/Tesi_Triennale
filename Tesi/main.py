# standard lib
import numpy as np
import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport
# Sci-kit functions
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# custom lib
import Data_shaping as sh
from ML_pipelines import learning, scoring
from Data_Aquirer import Aquirer
from threshold import air_cataloguer

###############################################################

# I import the data
df = Aquirer('https://bollettino.appa.tn.it/aria/opendata/csv/2021-02-23,2021-05-22/')
# df = pd.read_csv(r"C:\Users\beltr\PycharmProjects\Tesi\Air1.csv", encoding='latin-1', engine='python', decimal='.')
df2 = df.copy()  # copy of the dataset
# NB: each element in the column "Valore" is interpreted as a fucking str, since i have to change it
df["Valore"] = pd.to_numeric(df['Valore'], errors='coerce')
df = df.dropna()
# I prepare my data in a useful way, read the function description for further info
# list_stat is a dict!!np.nan
list_stat, stat_names = sh.Station_Classifier(df)

# I normalize the U.M. of the densities, thus uniforming them to the value of ug/cm
for col in stat_names:
    sh.UM_Normalizer(list_stat[col.item()])

# I assign each pollutant level to the respective measurment
pollutants = ["CO", "NO2", "PM10", "PM2_5", "O3", "SO2"]
# first i perform a subroutine to get the right acronym to run efficiently the main programme
for col in stat_names:
    list_stat[col.item()] = air_cataloguer(df=list_stat[col.item()],
                                           pollutants=sh.pollutants_acronym(sh.pollutants_names(list_stat[col.item()])))

# I perform a first level EDA
profiles = {}
# for col in stat_names:
# profiles[col.item()] = ProfileReport(list_stat[col.item()], title=col.item())
# profiles[col.item()].to_file("Profile_{0}.html".format(col.item()))
# print("Questa stazione è:", col.item())
# print(list_stat[col.item()])

# I prepare my data in a suitable form for ML
target = {}
for col in stat_names:
    target[col.item()] = list_stat[col.item()].loc[:, "Livello Inquinanti"]

# dataset to be used in ML
X = {}
y = {}
# I split train/test
print("YOLO")
for col in stat_names:
    X[col.item() + "train"], X[col.item() + "test"], y[col.item() + "train"], y[col.item() + "test"] = train_test_split(
        list_stat[col.item()].loc[:, list_stat[col.item()].columns != 'Livello Inquinanti'], target[col.item()],
        test_size=0.25)

# I eliminate incidental NaN
for col in stat_names:
    if (X[col.item() + "test"].isnull().any().any())==True:
        print(col.item())
        print(X[col.item() + "test"].isnull().any())


# let's start with the first algorithm Random Forest
RFC = {}
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
for col in stat_names:
    RFC[col.item()] = learning(X_train=X[col.item() + "train"], y_train=enc.fit_transform(y[col.item() + "train"].values[:,None]),
                               alg_name=RandomForestClassifier(oob_score=True))
    print("Per la stazione ", col.item(), "il risultato del training è:")
    scoring(X_test=X[col.item() + "test"], y_test=enc.fit_transform(y[col.item() + "test"].values[:,None]),
            pipe_fit=RFC[col.item()])
