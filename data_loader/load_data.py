import pandas as pd
import numpy as np
import datetime
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

def load_clean_nyc(path):
    df = load_dataset(path)
    y = df["dep_delay"]
    df = df[df["flight"] != "x"]
    df = df.dropna()
    df = df.drop(["year", "time_hour",  "dest", "carrier", "arr_time","dep_time", "sched_dep_time", "arr_time", "sched_arr_time", "air_time", "tailnum", "origin"], axis = 1)
    month_prefix = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    df["month_to_day"]=month_prefix[df["month"].tolist()]
    df["julian_datetime"]=df["month_to_day"]+df["day"]+df["hour"]/24+df["minute"]/1440
    df = df.drop(["month_to_day", "month",  "day", "hour", "minute"], axis = 1)
    return df
