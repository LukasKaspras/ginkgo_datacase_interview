import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

def pull_data():
    '''
    Pulls the raw "pump_sensor" data from AWS and return it as a pandas DataFrame

            Returns:
                pandas.DataFrame

    '''
    resp = urlopen("https://ga-data-cases.s3.eu-central-1.amazonaws.com/pump_sensor.zip")

    zipfile = ZipFile(BytesIO(resp.read()))

    # get the csv file name
    fname = zipfile.namelist()[0]

    # convert to pandas dateframe
    df = pd.read_csv(zipfile.open(fname), dtype=object)

    # close zipfile we don't need
    zipfile.close()

    return df