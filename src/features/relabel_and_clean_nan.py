import pandas as pd
import numpy as np
from helper import *

def give_danger_bool(df, danger_window):
    danger_window *= 1440
    last_rows_are_no_danger_window = df.index < (df.tail(1).index.values[0] - np.timedelta64(danger_window, 'm'))
    return (df.minutes_until_broken < danger_window) & (df.minutes_until_broken != 0) & last_rows_are_no_danger_window

def relabel_df(df):

    # We create a label column for regression.
    df['minutes_until_broken'] = df.groupby((df["machine_status"] == "BROKEN").cumsum()).cumcount(ascending=False)

    # We create an ln label column for regression
    df['ln_minutes_until_broken'] = np.log(df['minutes_until_broken'])
    # In order to not having to work with -inf later on.
    df['ln_minutes_until_broken'].replace(-np.inf, 0, inplace=True)

    for danger_window in give_binary_file_name_dict().keys():
        status_col, code_col, binary_status_col = give_danger_strings(danger_window)

        danger_bool = give_danger_bool(df, danger_window)
        df[status_col] = df.machine_status
        df.loc[danger_bool, status_col] = "DANGER"

        df[binary_status_col] = df[status_col].eq("DANGER", fill_value=0)

    return df

def clean_missing_values(df):
    sensor_cols = [col for col in df.columns if "sensor_" in col]

    sensors_with_few_missing_values = df[sensor_cols].isnull().sum()[df[sensor_cols].isnull().sum() < 100].index
    sensors_with_many_missing_values = df[sensor_cols].isnull().sum()[df[sensor_cols].isnull().sum() >= 100].index

    # We remove the rows where only few observations are missing.
    df = df.dropna(subset=sensors_with_few_missing_values)

    # We transform the columns with many missing values into boolean columns
    # that represent the presence of missing values.
    df[sensors_with_many_missing_values] = df[sensors_with_many_missing_values].isnull()

    return df

def save_cleaned_relabeled_df(path):

    path += "/interim/"
    df = read_raw_df()
    df = relabel_df(df)
    df = clean_missing_values(df)

    df.to_csv(path + "cleaned_relabeled_data.csv")

def save_dataframes_to_csv(path):
    df = pd.read_csv(path + "\\interim\\cleaned_relabeled_data.csv")

    path += "/processed/"

    sensor_cols = [col for col in df.columns if "sensor_" in col]

    file_name_regression = "regression_df.csv"
    file_name_ln_regression = "ln_regression_df.csv"

    binary_dict = give_binary_file_name_dict()
    multiclass_dict = give_binary_file_name_dict()


    df[sensor_cols + ["minutes_until_broken"]].to_csv(path + file_name_regression)
    df[sensor_cols + ["ln_minutes_until_broken"]].to_csv(path + file_name_ln_regression)

    for danger_window in give_binary_file_name_dict().keys():

        status_col, code_col, binary_status_col = give_danger_strings(danger_window)

        df[sensor_cols + [status_col]].to_csv(path + multiclass_dict[danger_window])
        df[sensor_cols + [binary_status_col]].to_csv(path + binary_dict[danger_window])



if __name__ == '__main__':
    project_dir = str(Path(__file__).resolve().parents[2])
    save_cleaned_relabeled_df(project_dir)
    save_dataframes_to_csv(project_dir)

