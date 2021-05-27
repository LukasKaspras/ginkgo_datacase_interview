import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

large = 22;
med = 16;
small = 12
params = {'axes.titlesize':   large,
          'legend.fontsize':  med,
          'figure.figsize':   (16, 10),
          'axes.labelsize':   med,
          'axes.titlesize':   med,
          'xtick.labelsize':  med,
          'ytick.labelsize':  med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
plt.tight_layout()

df = pd.read_csv("pump_sensor.csv")
df.timestamp = pd.to_datetime(df.timestamp)
df.set_index("timestamp", inplace=True)
df.drop(["sensor_15"], axis=1, inplace=True)

df['minutes_until_broken'] = df.groupby((df["machine_status"] == "BROKEN").cumsum()).cumcount(ascending=False)

last_rows_are_no_danger_window = df.index < (df.tail(1).index.values[0] - np.timedelta64(4 * 1440, 'm'))
# last_rows_are_no_danger_window = df.index < df[df["machine_status"] == "BROKEN"].tail(1).index

danger_bool = (df.minutes_until_broken < 4 * 1440) & (df.minutes_until_broken != 0) & last_rows_are_no_danger_window

df["machine_status_multiclass_danger_4"] = df["machine_status"].copy(deep=True)






def create_labelgraph_allLabels():
    status_dict = {"NORMAL": 2, "BROKEN": 0, "RECOVERING": 1, "DANGER": 0.0}

    df["code_col"] = df["machine_status_multiclass_danger_4"].map(status_dict)

    fig, ax = plt.subplots(figsize=(17, 2), dpi=1000)

    df.loc[danger_bool, "machine_status_multiclass_danger_4"] = "DANGER"
    failures = df[df["machine_status_multiclass_danger_4"] == "BROKEN"].index

    col = "sensor_02"

    df[col].plot(ax=ax)

    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  df["code_col"].values[np.newaxis],
                  cmap='RdYlGn', alpha=0.3)

    for location in ["left", "right", "bottom", "top"]:
        ax.spines[location].set_visible(False)

    for failure in failures:
        ax.axvline(failure, color='r', linewidth=2, alpha=0.7)

    fontdict = {'color': '#676C73'}
    ax.set_ylabel(str(col), fontdict=fontdict)
    ax.set(xlabel=None)

    for tick in ax.get_xticklabels():
        tick.set_color("#676C73")

    for tick in ax.get_yticklabels():
        tick.set_color("#676C73")

    plt.savefig("labelgraph_allLabels.png")

def create_labelgraph_binaryLabels():
    status_dict = {"NORMAL": 2, "BROKEN": 2, "RECOVERING": 2, "DANGER": 0.0}

    df["code_col"] = df["machine_status_multiclass_danger_4"].map(status_dict)

    fig, ax = plt.subplots(figsize=(17, 2), dpi=1000)

    df.loc[danger_bool, "machine_status_multiclass_danger_4"] = "DANGER"
    failures = df[df["machine_status_multiclass_danger_4"] == "BROKEN"].index

    col = "sensor_02"

    df[col].plot(ax=ax)

    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  df["code_col"].values[np.newaxis],
                  cmap='RdYlGn', alpha=0.3)

    for location in ["left", "right", "bottom", "top"]:
        ax.spines[location].set_visible(False)


    fontdict = {'color': '#676C73'}
    ax.set_ylabel(str(col), fontdict=fontdict)
    ax.set(xlabel=None)

    for tick in ax.get_xticklabels():
        tick.set_color("#676C73")

    for tick in ax.get_yticklabels():
        tick.set_color("#676C73")

    plt.savefig("labelgraph_binaryLabels.png")

def create_labelgraph_baseLabels():
    status_dict = {"NORMAL": 2, "BROKEN": 0, "RECOVERING": 1, "DANGER": 0.0}

    df["code_col"] = df["machine_status_multiclass_danger_4"].map(status_dict)

    fig, ax = plt.subplots(figsize=(17, 2), dpi=1000)

    failures = df[df["machine_status_multiclass_danger_4"] == "BROKEN"].index

    col = "sensor_02"

    df[col].plot(ax=ax)

    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  df["code_col"].values[np.newaxis],
                  cmap='RdYlGn', alpha=0.3)

    for location in ["left", "right", "bottom", "top"]:
        ax.spines[location].set_visible(False)

    for failure in failures:
        ax.axvline(failure, color='r', linewidth=2, alpha=0.7)

    fontdict = {'color': '#676C73'}
    ax.set_ylabel(str(col), fontdict=fontdict)
    ax.set(xlabel=None)

    for tick in ax.get_xticklabels():
        tick.set_color("#676C73")

    for tick in ax.get_yticklabels():
        tick.set_color("#676C73")

    plt.savefig("labelgraph_baseLabels.png")