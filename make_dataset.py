import pandas as pd
import numpy as np
import geopandas as gpd
import os

from pathlib import Path

"""
Script filled with functions useful towards the importation and creation of datasets
"""

##Paths per muoversi in cookiecutter

data_path_in = Path('data/raw')
data_path_pollutants = Path('data/raw/Inquinanti')
data_path_meteo = Path('data/raw/Meteo')

data_path_out = Path('data/processed')

#models_path = Path('models')

files = {'grid':['trentino_grid.geojson',"geojson"],
        'rain':['Pioggia.csv',"csv"],
        'speed':['Velocità.csv',"csv"],
        'humidity':['Umidità.csv',"csv"],
        'radiation':['Radiazione.csv',"csv"],
        'direction':['Direzione.csv',"csv"],
        'pressure':['Pressione.csv',"csv"],
        'temperature':['Temperatura.csv',"csv"],
        'path_pollutants': data_path_pollutants,
        'path_meteo': data_path_meteo
        }


def safe_import(inp):
    """
    Function that imports data from a file, turns it into a pandas dataframe,
    and prints the types of every variable to check for correctness of import
    To be used appropriately inside a notebook
    """
    filename = files[inp][0]
    filetype = files[inp][1]

    fl = data_path_in / filename
    if (filetype == "geojson"):
        out = gpd.read_file(fl)
    if (filetype == "csv"):
        out = pd.read_csv(fl)
    if (filetype == "json"):
        out = pd.read_json(fl, orient="values")
    if (filetype == "shape"):
        out = gpd.read_file(fl)
    print("SafeImport_Output:  ", out.keys())

    return out

def list_in_directory(mypath):
    """
    This function acquires the name of all the files in a given directory. it returns a list
    """
    arr = os.listdir(files[mypath])
    return arr


def acquirer(file):
    """
    This function aquires the data from the server, data must be a csv format
    """
    data = pd.read_csv(file, encoding='latin-1', engine='python', decimal='.')
    data.columns = data.columns.str.replace('[()]', '', regex=True)
    data = data.replace('[()]', '', regex=True)
    data.dropna(inplace=True)
    return data

def Aquirer_meteo(data_path, inp):
    """
    This function gets data in csv format and cleans them from thing that make the alg crash.
    It is specifically made to acquire data nested in folders and sub folders.
    It needs a path as an argument (Better given using pathlib.Path)
    """
    filename = files[inp][0]

    fl = data_path / filename
    data = pd.read_csv(fl, encoding='latin-1', engine='python', decimal='.')
    data.columns = data.columns.str.replace('[()]', '', regex=True)
    data = data.replace('[()]', '', regex=True)
    data.dropna(inplace = True)

    labels = data.columns
    print("You just added this feature ", labels[1], "to the DF")
    return data


def appforth(df, line):
    """
    Function that adds a line at the top of a dataframe
    """
    df.loc[-1]=line
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index
    return df


def orderstation(weatherdf):
    """
    Funzione che ordina il dataframe del weather per estrarre 
    caratteristiche uniche delle stazioni, quali nome, posizione, elevazione
    Comodo quando devo trovare la stazione più vicina ad un punto
    """

    stazioni = weatherdf['station'].unique()
    coltmpr=["station", "elevation", "geometry"]

    station_stats=gpd.GeoDataFrame(columns=coltmpr)

    for idx,stat in enumerate(stazioni):
        temp = pd.DataFrame(weatherdf[weatherdf['station']==stat])
        station_stats.loc[idx]="NaN"
        
        station_stats.loc[idx]["geometry"]=temp.loc[temp.index[0],:]["geometry"]
        station_stats.loc[idx]["station"]=stat
        station_stats.loc[idx]["elevation"]=temp.loc[temp.index[0],:]["elevation"]
    return station_stats


def pollutants_acronym(pollutants):
    """
    It change the names of the pollutants to its acronym, it gives back a string
    """
    list_pollutants = []
    for element in pollutants:
        list_pollutants.append(etiquette(element))
    return list_pollutants


def etiquette(element):
    """
    It change the element of the string to its acronym, it return a char
    """
    if element == 'PM10':
        return "PM10"
    elif element == 'Biossido di Azoto':
        return "NO2"
    elif element == 'Ozono':
        return "O3"
    elif element == 'PM2.5':
        return "PM2_5"
    elif element == 'Ossido di Carbonio':
        return "CO"
    elif element == 'Biossido Zolfo':
        return "SO2"
    else:
        print("Something went wrong in classifying pollutant level!")

def pollutants_names(df):
    """
    It extract the names of the matrix from the dataset
    """
    poll_names = pd.DataFrame(df['Inquinante'].unique()).to_numpy()
    return poll_names



def UM_Normalizer(df):
    """
    It normalize the data in order to have the same U.M.
    """
    df[df['Unità di misura'].str.contains('mg/mc')].loc[:, 'Valore'].multiply(1000)
    df.drop(labels="Unità di misura", axis="columns", inplace=True)


varconv={0 : "temperatures.", 1 : "precipitations."}
def find_Weather(weatherdf, month, day, hour, stationName, varType=0):
    """
    Funzione che cerca il valore dentro al database weather dato, per una certa data+ora fornita
    Gli input sono autoesplicativi, varType=0 vuol dire temperaura, varType=1 vuol dire precipitation
    """

    cellname="%02d%02d" % (int(np.floor(hour)),int((hour%1)*60))
    cellname=varconv[varType]+cellname

    df=weatherdf[weatherdf['station']==stationName]
    df=df[ df['date']==("2013-%02d-%02d"%(month,day)) ]

    #Se manca il dato posso procedere in 2 modi
        #1) Lo prendo mezz'ora prima o dopo che non varia troppo (Operazione non così banale e unsafe)
        #2) ritorna NaN, avrò meno statistica nell'EDA ma non importa, i NaN son pochi
        # In questa versione, uso la 2
    if(df[cellname].isnull().all()):
        return np.NAN
    return float(df[cellname])


def scale(v):
    """
    Funzione che scala un vettore, ovvero sposta l'i-esimo elemento all'i+1-esimo indice
    Elimina l'ultimo elemento della sequenza, e mette -1000 nel primo (per i nostri scopi è comodo così)
    Ritorna il vettore scalato
    """
    v.insert(0, -1000)       #Insert front
    del v[-1]                #Pop back
    return v

def temporal_shift(dict, keys):
    """
    Funzione che tratta i dataframe raffinati per produrre un nuovo dataframe atto a fare il machine learning
    Ritorna il dataframe stesso
    Si basa sull'avere i databases nella cartella processed quindi fare attenzione
    """
    columns = {}
    for i in keys:
        columns[i] = dict[i].columns
    label_add = [" -1h"," -2h", " -3h", " -4h"]
    label_shifted = ["", " -1h"," -2h", " -3h"]

    for i in keys:
        for k in columns[i]:
            counter = 0
            if (k != "Year" and k != "Hour" and k != "Month" and k != "Day"):
                for j in label_add:
                    h = label_shifted[counter]
                    dict[i][k + j] = scale(list(dict[i].loc[:,k + h]))
                    counter = counter + 1
    return dict


def corr_shift(dict, keys):
    """
    Funzione che tratta i dataframe raffinati per produrre un nuovo dataframe atto a fare il machine learning
    Ritorna il dataframe stesso
    Si basa sull'avere i databases nella cartella processed quindi fare attenzione
    """
    columns = {}
    for i in keys:
        columns[i] = dict[i].columns
    label_add = [" -1h"," -2h", " -3h", " -4h"," -5h", " -6h", "- 7h", "- 8h", "- 9h", "- 10h", "- 11h", "- 12h",
                 "- 13h", "- 14h", "- 15h", "- 16h", " - 17h", "- 18h", " -19h", " -20h", " -21h", " -22h", " -23h",
                 " -24h"]
    label_shifted = ["", " -1h"," -2h", " -3h"," -4h"," -5h", " -6h", "- 7h", "- 8h", "- 9h", "- 10h", "- 11h","- 12h",
                 "- 13h", "- 14h", "- 15h","- 16h", " - 17h", "- 18h", " -19h", " -20h", " -21h", " -22h", " -23h"]

    for i in keys:
        for k in columns[i]:
            if k == "O3":
                counter = 0
                if (k != "Year" and k != "Hour" and k != "Month" and k != "Day"):
                    for j in label_add:
                        h = label_shifted[counter]
                        dict[i][k + j] = scale(list(dict[i].loc[:,k + h]))
                        counter = counter + 1
    return dict





