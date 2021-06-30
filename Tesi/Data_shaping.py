import pandas as pd

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
