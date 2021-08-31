import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
data_path_in = Path('./../data/raw')

data_path_out = Path('./../data/processed')

models_path = Path('./../models')


"""
Script filled with functions useful towards the importation and creation of datasets
Davide Micheli helped in the creation of this library
Author: Lorenzo Beltrame - IZerab
"""
# in this section there are the functions that modify the dataset in a more suitable form


def UM_Normalizer(df):
    """
    It normalize the data in order to have the same U.M.
    """
    df[df['Unità di misura'].str.contains('mg/mc')].loc[:, 'Valore'].multiply(1000)
    df.drop(labels="Unità di misura", axis="columns", inplace=True)


def pollutants_names(df):
    """
    It extract the names of the matrix from the dataset
    """
    poll_names = pd.DataFrame(df['Inquinante'].unique()).to_numpy()
    return poll_names


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


def pollutants_acronym(pollutants):
    """
    It change the names of the pollutants to its acronym, it gives back a string
    """
    list_pollutants = []
    for element in pollutants:
        list_pollutants.append(etiquette(element))
    return list_pollutants


def Station_Classifier(df):
    """
    This function subdivides the raw data in datasets labeled by the relative station name, it returns a dict and the
    string with the names of the stations
    """
    # I generate a dict in which each cell is a Dataframe relative to a station
    stat_names = pd.DataFrame(df['Stazione'].unique()).to_numpy()
    stations = {}
    for col in stat_names:
        stations[col.item()] = df[df['Stazione'].str.contains(col.item())]
        stations[col.item()].drop('Stazione', axis='columns', inplace=True)
    return stations, stat_names


# This is a much more compact version but will work only for python 3.10 +
# Fuction working for python 3.9 is found above
# def etiquette(element):
#    """
#    It change the element of the string to its acronym, it return a char
#    """
#    match element:
#        case 'PM10':
#            return "PM10"
#        case 'Biossido di Azoto':
#            return "NO2"
#        case 'Ozono':
#            return "O3"
#        case 'PM2.5':
#            return "PM2_5"
#        case 'Ossido di Carbonio':
#            return "CO"
#        case 'Biossido Zolfo':
#            return "SO2"




#################### DATA FROM PREVIOUS EXPERIENCES, WIP ################
##Paths per muoversi in cookiecutter


files = {'grid': ['trentino-grid.geojson', "geojson"],
         'adm_reg': ['administrative_regions_Trentino.json', "json"],
         'weather': ['meteotrentino-weather-station-data.json', "json"],
         'precip': ['precipitation-trentino.csv', "csv"],
         'precip-avail': ['precipitation-trentino-data-availability.csv', "csv"],
         'SET-1': ['SET-nov-2013.csv', "csv"],
         'SET-2': ['SET-dec-2013.csv', "csv"],
         'SET-lines': ['line.csv', "csv"],
         'twitter': ['social-pulse-trentino.geojson', "json"],
         'regions': ['Com01012013/Com01012013_WGS84.shp', "shape"],
         'circoscrizioni': ['CircoscrizioniTN/circoscrizioni.shp', "shape"]}


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


def appforth(df, line):
    """
    Function that adds a line at the top of a dataframe
    """
    df.loc[-1] = line
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
    coltmpr = ["station", "elevation", "geometry"]

    station_stats = gpd.GeoDataFrame(columns=coltmpr)

    for idx, stat in enumerate(stazioni):
        temp = pd.DataFrame(weatherdf[weatherdf['station'] == stat])
        station_stats.loc[idx] = "NaN"

        station_stats.loc[idx]["geometry"] = temp.loc[temp.index[0], :]["geometry"]
        station_stats.loc[idx]["station"] = stat
        station_stats.loc[idx]["elevation"] = temp.loc[temp.index[0], :]["elevation"]
    return station_stats


varconv = {0: "temperatures.", 1: "precipitations."}


def find_Weather(weatherdf, month, day, hour, stationName, varType=0):
    """
    Funzione che cerca il valore dentro al database weather dato, per una certa data+ora fornita
    Gli input sono autoesplicativi, varType=0 vuol dire temperaura, varType=1 vuol dire precipitation
    """

    cellname = "%02d%02d" % (int(np.floor(hour)), int((hour % 1) * 60))
    cellname = varconv[varType] + cellname

    df = weatherdf[weatherdf['station'] == stationName]
    df = df[df['date'] == ("2013-%02d-%02d" % (month, day))]

    # Se manca il dato posso procedere in 2 modi
    # 1) Lo prendo mezz'ora prima o dopo che non varia troppo (Operazione non così banale e unsafe)
    # 2) ritorna NaN, avrò meno statistica nell'EDA ma non importa, i NaN son pochi
    # In questa versione, uso la 2
    if (df[cellname].isnull().all()):
        return np.NAN
    return float(df[cellname])


def Wday(month, day):
    """
    Function that yields the weekday given the month and the day
        Works for the year 2013, months 11 and 12
    """
    out = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    if (month == 11):
        return out[(4 + day) % 7]
    if (month == 12):
        return out[(6 + day) % 7]


def scale(v):
    """
    Funzione che scala un vettore, ovvero sposta l'i-esimo elemento all'i+1-esimo indice
    Elimina l'ultimo elemento della sequenza, e mette -1000 nel primo (per i nostri scopi è comodo così)
    Ritorna il vettore scalato
    """
    v.insert(0, -1000)  # Insert front
    del v[-1]  # Pop back
    return v


def df_reg():
    """
    Funzione che tratta i dataframe raffinati per produrre un nuovo dataframe atto a fare il machine learning
    Ritorna il dataframe stesso
    Si basa sull'avere i databases nella cartella processed quindi fare attenzione
    """
    dfTweets = pd.read_csv(data_path_out / "twitter_final.csv")
    dfTemp = pd.read_csv(data_path_out / "weather_final.csv")
    dfElectric = pd.read_csv(data_path_out / "electro_final.csv")

    columnsDay = ["Tweet1m", "Tweet2m", "Tavg1m", "Tavg2m", "Rainmax1m", "Rainmax2m",
                  "Rainavg1m", "Rainavg2m", "Electro1m", "Electro2m"]
    columnsNight = ["Tweet1n", "Tweet2n", "Tavg1n", "Tavg2n", "Rainmax1n", "Rainmax2n",
                    "Rainavg1n", "Rainavg2n", "Electro1n", "Electro2n"]
    columns = columnsDay + columnsNight + ["Weekday", "TargetDay", "TargetNight"]

    out = pd.DataFrame(columns=columns)

    ##Tweets:
    # NOTA: per ora il numero di tweets non sarà normalizzato ai giorni della settimana
    TwDay = dfTweets[dfTweets["hours"] > 7.9]
    TwDay = TwDay[TwDay["hours"] < 18.9]

    # Questo autosorta e raggruppa per day
    NtwDay = pd.DataFrame({'Counts': TwDay.groupby(
        ['month', 'day']).size()}).reset_index()
    out["TargetDay"] = NtwDay["Counts"]

    TwNight = dfTweets[dfTweets["hours"] > 18.9]

    NtwNight = pd.DataFrame({'Counts': TwNight.groupby(
        ['month', 'day']).size()}).reset_index()
    out["TargetNight"] = NtwNight["Counts"]

    # Per l'input dei twwets dei giorni prima, scorro il vettore inputs
    temp = list(NtwDay["Counts"])
    out["Tweet1m"] = scale(temp)
    out["Tweet2m"] = scale(temp)

    temp = list(NtwNight["Counts"])
    out["Tweet1n"] = scale(temp)
    out["Tweet2n"] = scale(temp)

    # Weekday
    """Note: it may be best to normalize the number of tweets to the 
        individual weekday (as in: renormalizing a temporal serie); 
        we will treat the weekday as a feature which should be roughly equivalent"""
    temp = []
    for i in [11, 12]:
        for j in range(0, 19 + i):
            temp.append(Wday(i, j))
    out["Weekday"] = temp

    # Average temperature
    # The average is computer over all the stations
    # DAY
    colTempDay = ["date", "temperatures.0900", "temperatures.0915", "temperatures.0930", "temperatures.0945"] + \
                 ["temperatures." + str(int(1000 + 100 * np.floor(i / 4) + (i % 4) * 15)) for i in range(0, 36)]
    # Medio sulle stazioni
    Tavg = pd.DataFrame(data=dfTemp, columns=colTempDay).groupby("date").mean()
    # Medio sulla giornata, che qua è rappresentata dalle colonne
    Tavg = list(Tavg.swapaxes(0, 1).mean())

    out["Tavg1m"] = scale(Tavg)
    out["Tavg2m"] = scale(Tavg)

    # NIGHT
    colTempNight = ["date"] + ["temperatures." + str(int(1900 + 100 * np.floor(i / 4) + (i % 4) * 15)) for i in
                               range(0, 20)]
    Tavg = pd.DataFrame(data=dfTemp, columns=colTempNight).groupby("date").mean()
    Tavg = list(Tavg.swapaxes(0, 1).mean())

    out["Tavg1n"] = scale(Tavg)
    out["Tavg2n"] = scale(Tavg)

    # Similarly, precipitation, let's also fetch the maximum
    # DAY
    colPrecDay = ["date", "precipitations.0900", "precipitations.0915", "precipitations.0930", "precipitations.0945"] + \
                 ["precipitations." + str(int(1000 + 100 * np.floor(i / 4) + (i % 4) * 15)) for i in range(0, 36)]
    Pavg = pd.DataFrame(data=dfTemp, columns=colPrecDay).groupby("date").mean()
    TopP = pd.DataFrame(data=dfTemp, columns=colPrecDay).groupby("date").max()
    Pavg = list(Pavg.swapaxes(0, 1).mean())
    TopP = list(TopP.swapaxes(0, 1).mean())

    out["Rainavg1m"] = scale(Pavg)
    out["Rainavg2m"] = scale(Pavg)
    out["Rainmax1m"] = scale(TopP)
    out["Rainmax2m"] = scale(TopP)

    # NIGHT
    colPrecNight = ["date"] + ["precipitations." + str(int(1900 + 100 * np.floor(i / 4) + (i % 4) * 15)) for i in
                               range(0, 20)]
    Pavg = pd.DataFrame(data=dfTemp, columns=colPrecNight).groupby("date").mean()
    TopP = pd.DataFrame(data=dfTemp, columns=colPrecDay).groupby("date").max()
    Pavg = list(Pavg.swapaxes(0, 1).mean())
    TopP = list(TopP.swapaxes(0, 1).mean())

    out["Rainavg1n"] = scale(Pavg)
    out["Rainavg2n"] = scale(Pavg)
    out["Rainmax1n"] = scale(TopP)
    out["Rainmax2n"] = scale(TopP)

    # Electricity
    ElDay = dfElectric[dfElectric["hours"] > 7.9]
    ElDay = ElDay[ElDay["hours"] < 18.9]

    ElNight = dfElectric[dfElectric["hours"] > 18.9]

    # Vogliamo consumo (in amp) attraverso tutta la giornata
    # Dati lasciati a leggera interpretazione, credo che sommare su tutte le lines dia consumo netto territorio
    # Qui facciamo quello
    ElDayT = list(ElDay.groupby(["month", "day"])["Value Amp"].sum())
    ElNightT = list(ElNight.groupby(["month", "day"])["Value Amp"].sum())

    out["Electro1m"] = scale(ElDayT)
    out["Electro2m"] = scale(ElDayT)
    out["Electro1n"] = scale(ElNightT)
    out["Electro2n"] = scale(ElNightT)

    out.drop(index=[0, 1], inplace=True)
    out.reset_index(inplace=True)
    out.drop(columns="index", inplace=True)
    return out





