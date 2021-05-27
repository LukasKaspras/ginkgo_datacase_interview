from pathlib import Path
import pull_sensor_data

def main(path):
    pass

if __name__ == '__main__':
    project_dir = str(Path(__file__).resolve().parents[2])

    file_name_raw_data = "/pump_censor.csv"
    path_interim = project_dir + "/data"

    main(path)

    print(project_dir)