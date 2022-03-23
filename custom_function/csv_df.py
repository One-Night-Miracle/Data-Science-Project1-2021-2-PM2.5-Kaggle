# Import the libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def toDF(pm25_filename, temp_filename, wind_filename):
    pm25_df = pd.read_csv('datasci_dataset_2022/' +
                          pm25_filename, names=['Time', 'PM25'], skiprows=1)
    pm25_df['Time'] = pd.to_datetime(pm25_df['Time'])
    # pm25_df['Time'] = pm25_df['Time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')
    # pm25_df['Time'] = pm25_df['Time'].dt.tz_localize(None)
    pm25_df.set_index('Time', inplace=True)
    pm25_df.columns = ['PM25']
    pm25_df = pm25_df[~pm25_df.index.duplicated(keep='first')]
    pm25_df.interpolate(inplace=True)
    pm25_df.index = pd.DatetimeIndex(pm25_df.index)

    temp_df = pd.read_csv('datasci_dataset_2022/' +
                          temp_filename, names=['Time', 'Temp'], skiprows=1)
    temp_df['Time'] = pd.to_datetime(temp_df['Time'])
    temp_df.set_index(temp_df['Time'], inplace=True)
    temp_df.drop(columns={'Time'}, inplace=True)
    temp_df.columns = ['Temp']
    # pad() is similar to fillna() with forward filling
    temp_df = temp_df.resample('h').pad().ffill()
    temp_df = temp_df.bfill()
    # temp_df.index = pd.DatetimeIndex(temp_df.index)

    wind_df = pd.read_csv('datasci_dataset_2022/'+wind_filename,
                          names=['Time', 'WindSpeed', 'WindDir'], skiprows=1)
    wind_df['Time'] = pd.to_datetime(wind_df['Time'])
    wind_df.set_index(wind_df['Time'], inplace=True)
    wind_df.drop(columns={'Time'}, inplace=True)
    wind_df.columns = ['WindSpeed', 'WindDir']
    # forward filling
    wind_df = wind_df.resample('h').ffill()
    wind_df = wind_df.bfill()
    # wind_df.index = pd.DatetimeIndex(wind_df.index)

    pm25_df['copy_index'] = pm25_df.index
    df = pd.merge(pm25_df, temp_df, left_index=True, right_index=True)
    df = pd.merge(df, wind_df, left_index=True, right_index=True)

    df = df[['Temp', 'WindSpeed', 'WindDir', 'PM25']]

    # padding first and last indices
    # df = df.ffill()
    # df = df.bfill()

    return df


def toDFtest(pm25_filename, temp_filename, wind_filename):
    pm25_df = pd.read_csv('datasci_dataset_2022/' +
                          pm25_filename, names=['Time', 'PM25'], skiprows=1)
    pm25_df['Time'] = pd.to_datetime(pm25_df['Time'])
    pm25_df.set_index('Time', inplace=True)
    pm25_df.columns = ['PM25']
    pm25_df = pm25_df[~pm25_df.index.duplicated(keep='first')]
    pm25_df.interpolate(inplace=True)
    pm25_df.index = pd.DatetimeIndex(pm25_df.index)

    temp_df = pd.read_csv('datasci_dataset_2022/' +
                          temp_filename, names=['Time', 'Temp'], skiprows=1)
    temp_df['Time'] = pd.to_datetime(temp_df['Time'])
    temp_df.set_index(temp_df['Time'], inplace=True)
    temp_df.drop(columns={'Time'}, inplace=True)
    temp_df.columns = ['Temp']
    temp_df = temp_df.resample('h').ffill()
    temp_df = temp_df.bfill()

    wind_df = pd.read_csv('datasci_dataset_2022/'+wind_filename,
                          names=['Time', 'WindSpeed', 'WindDir'], skiprows=1)
    wind_df['Time'] = pd.to_datetime(wind_df['Time'])
    wind_df.set_index(wind_df['Time'], inplace=True)
    wind_df.drop(columns={'Time'}, inplace=True)
    wind_df.columns = ['WindSpeed', 'WindDir']
    # backward filling
    wind_df = wind_df.resample('h').ffill()
    wind_df = wind_df.bfill()
    # wind_df.index = pd.DatetimeIndex(wind_df.index)

    pm25_df['copy_index'] = pm25_df.index
    df = pd.merge(pm25_df, temp_df, left_index=True, right_index=True)
    df = pd.merge(df, wind_df, left_index=True, right_index=True)

    df = df[['Temp', 'WindSpeed', 'WindDir', 'PM25']]

    return df
