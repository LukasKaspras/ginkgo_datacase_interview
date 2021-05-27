# -*- coding: utf-8 -*-
from pathlib import Path
import relabel_and_clean_nan

def main(path):
    '''
    Pulls raw data and saves it as csv under the given path.

            Arguments:
                path (str) - the location where the csv is to be saved
    '''
    relabel_and_clean_nan.save_cleaned_relabeled_df(path)
    relabel_and_clean_nan.save_dataframes_to_csv(path)


if __name__ == '__main__':
    project_dir = str(Path(__file__).resolve().parents[2])
    
    path = project_dir + "/data"

    main(path)

    print(project_dir)
