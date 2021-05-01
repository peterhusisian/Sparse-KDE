import pandas as pd

def load_dataset(path):
    df=pd.read_csv(path, sep=',',header=0)
    return df


def load_clean_alfredo_saurez(path, airline_name):
    df = load_dataset(path)
    df = df.replace('-', '', regex=True)
    df = df[df["sunshine-duration"] != -999]
    y = df["delayed"]
    X = df[df["airline"]==airline_name]
    #remove delayed for the y variable
    X = X.drop(["delayed", "year", "month", "destination", "id", "schedule", "departure", "airline", "snowfall-amount"], axis = 1)
    return X
