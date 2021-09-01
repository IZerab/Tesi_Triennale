import pandas as pd
#It aquires data from the trentino open data server, for A22 station it deletes parentheses!!
def acquirer(file):
    """
    This function aquires the data from the server, data must be a csv format
    """
    data = pd.read_csv(file, encoding='latin-1', engine='python', decimal='.')
    data.columns = data.columns.str.replace('[()]', '', regex=True)
    data = data.replace('[()]', '', regex=True)
    data.dropna(inplace=True)
    labels = data.columns
    print("The columns of the data set are: ", labels)
    return data