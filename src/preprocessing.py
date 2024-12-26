import pandas as pd
import numpy as np
import glob
import os
import random
import json 
from sklearn.preprocessing import StandardScaler


with open('./config/config.json', 'r') as file:
    config = json.load(file)


def load_data(path, k=100, seed=42):
    random.seed(seed)
    all_files = glob.glob(os.path.join(path, "*.csv"))
    samples = random.sample(all_files, k=k)

    li = []

    for i, filename in enumerate(samples):
        df = pd.read_csv(filename, sep=";")
        df["Sensor"] = i
        df = df.set_index(keys="Sensor", drop=True)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=False)

    return df

def preprocessing(df):
    # cyclic encoding
    df['MM_sin'] = np.sin(2 * np.pi * df['MM']/12.0)
    df['MM_cos'] = np.cos(2 * np.pi * df['MM']/12.0)

    df['DOY_sin'] = np.sin(2 * np.pi * df['DOY']/365.0)
    df['DOY_cos'] = np.cos(2 * np.pi * df['DOY']/365.0)

    # set index 
    df = df.reset_index()
    df = df.set_index([df["YYYY"].rename("Year"), df["MM"].rename("Month"), df["DD"].rename("Day"), df["Sensor"]], drop=True)
    df = df.drop(columns=["YYYY", "MM", "DD", "DOY", "Sensor"], axis=1)
    df = df.sort_index()

    # scale
    exclude_columns = ['MM_sin', 'MM_cos', 'DOY_sin','DOY_cos']
    # Columns to scale
    scale_columns = df.columns.difference(exclude_columns)
    scaler = StandardScaler()
    df[scale_columns] = scaler.fit_transform(df[scale_columns])

    df.to_csv("../output/data.csv", sep=";", index=True)

    return df


def load_preprocessed_data():
    df = pd.read_csv("./output/data.csv", sep=";",index_col=["Year", "Month", "Day", "Sensor"])
    return df


def run_preprocessing(k=100):
    df = load_data(config["data_path"], k)
    df = preprocessing(df)
    return df

if __name__ == "__main__":
    run_preprocessing()