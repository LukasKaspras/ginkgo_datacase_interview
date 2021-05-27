# -*- coding: utf-8 -*-
from pathlib import Path
import pull_sensor_data

def main(path):
    '''
    Pulls raw data and saves it as csv under the given path.

            Arguments:
                path (str) - the location where the csv is to be saved
    '''
    df = pull_sensor_data.pull_data()
    df.to_csv(path)


if __name__ == '__main__':
    project_dir = str(Path(__file__).resolve().parents[2])

    file_name_raw_data = "/pump_censor.csv"
    path = project_dir + "/data/raw" + file_name_raw_data

    main(path)

    print(project_dir)
