import glob
import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(path, k=100, seed=42):
    print(f'Loading data from {path} ...')
    random.seed(seed)
    all_files = glob.glob(os.path.join(path, "*.csv"))
    samples = random.sample(all_files, k=k)

    li = []
    successful_count = 0
    target_count = len(samples)

    i = 0
    while successful_count < target_count and i < len(samples):
        filename = samples[i]
        try:
            if successful_count % 5 == 0 and successful_count > 0:
                print(f'Loading {successful_count}th file...')
            df = pd.read_csv(filename, sep=";")
            df["Sensor"] = successful_count
            df = df.set_index(keys="Sensor", drop=True)
            li.append(df)
            successful_count += 1
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
        finally:
            i += 1

    df = pd.concat(li, axis=0, ignore_index=False)

    return df


def preprocessing(df):
    print('Performing cyclic encoding...')
    # cyclic encoding
    df['MM_sin'] = np.sin(2 * np.pi * df['MM'] / 12.0)
    df['MM_cos'] = np.cos(2 * np.pi * df['MM'] / 12.0)

    df['DOY_sin'] = np.sin(2 * np.pi * df['DOY'] / 365.0)
    df['DOY_cos'] = np.cos(2 * np.pi * df['DOY'] / 365.0)

    # set index
    df = df.reset_index()
    df = df.set_index([df["YYYY"].rename("Year"), df["MM"].rename("Month"), df["DD"].rename("Day"), df["Sensor"]],
                      drop=True)
    df = df.drop(columns=["MM", "DD", "DOY", "Sensor"], axis=1)
    df = df.sort_index()

    # scale
    exclude_columns = ['YYYY', 'MM_sin', 'MM_cos', 'DOY_sin', 'DOY_cos', 'prec']
    # Columns to scale
    scale_columns = df.columns.difference(exclude_columns)
    scaler = StandardScaler()
    df[scale_columns] = scaler.fit_transform(df[scale_columns])

    min_max_scaler = MinMaxScaler()
    df["YYYY"] = min_max_scaler.fit_transform(df["YYYY"].values.reshape(-1, 1))

    return df
