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


def build_danger_window_df(danger_window):
    '''Create a DataFrame with new Label for Rows that are in the danger_window before the machine breaks.
    machine_status_multiclass_danger_%s -- {"NORMAL", "BROKEN", "RECOVERING", "DANGER"}
    machine_status_binary_danger_%s -- 1 (= "DANGER"), 0 (= "NOT DANGER")

    Keyword arguments:
    danger_window -- length of the danger_window in days (1 -> 1440 Rows before "BROKEN" are labeled as "DANGER")
    binary -- True, Labels: 1 for Danger, 0 for Not Danger
              False, Labels: {"NORMAL", "BROKEN", "RECOVERING", "DANGER"}
    '''

    df = read_raw_df()
    sensor_cols = [col for col in df.columns if "sensor_" in col]

    file_name_binary = "binary_danger_window_%s_days_df.csv" % str((danger_window))
    file_name_multiclass = "multiclass_danger_window_%s_days_df.csv" % str((danger_window))

    temp_df = df.copy(deep=True)

    status_col = "machine_status_multiclass_danger_%s" % (str(danger_window))
    binary_status_col = "machine_status_binary_danger_%s" % (str(danger_window))

    danger_bool = give_danger_bool(temp_df, danger_window)
    temp_df.loc[danger_bool, 'machine_status'] = "DANGER"
    temp_df.rename(columns={'machine_status': status_col}, inplace=True)

    temp_df[binary_status_col] = temp_df[status_col].eq("DANGER", fill_value=0)

    # temp_df.to_csv(file_name)
    sensors_with_few_missing_values = df[sensor_cols].isnull().sum()[df[sensor_cols].isnull().sum() < 100].index
    sensors_with_many_missing_values = df[sensor_cols].isnull().sum()[df[sensor_cols].isnull().sum() >= 100].index

    # We remove the rows where only few observations are missing.
    temp_df = temp_df.dropna(subset=sensors_with_few_missing_values)

    # We transform the columns with many missing values into boolean columns
    # that represent the presence of missing values.
    temp_df[sensors_with_many_missing_values] = temp_df[sensors_with_many_missing_values].isnull()

    sensor_cols_subset = [col for col in sensor_cols if col in temp_df.columns]

    # We create two csv files:
    df_multiclass = temp_df[sensor_cols_subset + [status_col]]  # .to_csv(file_name_multiclass)
    df_binary = temp_df[sensor_cols_subset + [binary_status_col]]  # .to_csv(file_name_binary)

    return df_binary, df_multiclass, file_name_binary, file_name_multiclass


def build_regression_df():
    '''Create a DataFrame with new Label for Rows that are in the danger_window before the machine breaks.
    machine_status_multiclass_danger_%s -- {"NORMAL", "BROKEN", "RECOVERING", "DANGER"}
    machine_status_binary_danger_%s -- 1 (= "DANGER"), 0 (= "NOT DANGER")

    Keyword arguments:
    danger_window -- length of the danger_window in days (1 -> 1440 Rows before "BROKEN" are labeled as "DANGER")
    binary -- True, Labels: 1 for Danger, 0 for Not Danger
              False, Labels: {"NORMAL", "BROKEN", "RECOVERING", "DANGER"}
    '''

    df = read_raw_df()
    sensor_cols = [col for col in df.columns if "sensor_" in col]

    file_name_regression = "regression_df.csv"
    file_name_ln_regression = "ln_regression_df.csv"

    temp_df = df.copy(deep=True)

    # temp_df.to_csv(file_name)
    sensors_with_few_missing_values = df[sensor_cols].isnull().sum()[df[sensor_cols].isnull().sum() < 100].index
    sensors_with_many_missing_values = df[sensor_cols].isnull().sum()[df[sensor_cols].isnull().sum() >= 100].index

    # We remove the rows where only few observations are missing.
    temp_df = temp_df.dropna(subset=sensors_with_few_missing_values)

    # We transform the columns with many missing values into boolean columns
    # that represent the presence of missing values.
    temp_df[sensors_with_many_missing_values] = temp_df[sensors_with_many_missing_values].isnull()

    sensor_cols_subset = [col for col in sensor_cols if col in temp_df.columns]

    # We create two csv files:
    df_regression = temp_df[sensor_cols_subset + ["minutes_until_broken"]] # .to_csv(file_name_regression)
    df_ln_regression = temp_df[sensor_cols_subset + ["ln_minutes_until_broken"]] # .to_csv(file_name_ln_regression)

    return df_regression, df_ln_regression





if __name__ == '__main__':
    df = read_raw_df()
    df = relabel_df(df)
    print(df.columns, df.head())

print(give_binary_file_name_dict())
