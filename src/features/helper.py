import pandas as pd
from pathlib import Path

project_dir = str(Path(__file__).resolve().parents[2])

def give_binary_file_name_dict(danger_windows=range(0, 6)):
    binary_file_names = [give_file_name(i) for i in danger_windows]
    return dict(zip(danger_windows, binary_file_names))


def give_multiclass_file_name_dict(danger_windows=range(0, 6)):
    multiclass_file_names = [give_file_name(i, binary=False) for i in danger_windows]
    return dict(zip(danger_windows, multiclass_file_names))

def give_file_name(danger_window, binary=True):
    if binary:
        return "binary_danger_window_%s_days_df.csv" % str((danger_window))
    else:
        return "multiclass_danger_window_%s_days_df.csv" % str((danger_window))

def give_danger_strings(danger_window):
    status_col = "machine_status_danger_%s" % (str(danger_window))
    code_col = "code_danger_%s" % (str(danger_window))
    binary_status_col = "machine_status_binary_danger_%s" % (str(danger_window))

    return (status_col, code_col, binary_status_col)

def read_raw_df():
    df = pd.read_csv(project_dir + "\\data\\raw\\pump_censor.csv")
    df.timestamp = pd.to_datetime(df.timestamp)
    df.set_index("timestamp", inplace=True)
    df.drop(["sensor_15"], axis=1, inplace=True)
    return df