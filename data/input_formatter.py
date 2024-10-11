#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:11:10 2024

@author: goharshoukat
"""
import glob
import pandas as pd
import numpy as np

summary = pd.read_csv("summary.csv")
direc = "cpt_filtered_datasets/"
files = glob.glob(direc + "*.csv")
files = np.sort([x.replace("cpt_filtered_datasets/", "") for x in files])
location = pd.read_csv("location.csv", usecols=["CPT", "lat", "lng"])
tmp = ["0" + l for l in location.loc[:10, "CPT"] if float(l) < 10]
for i in range(len(tmp)):
    location.loc[i, "CPT"] = tmp[i]
location["CPT"] = ["CPT_" + location for location in location.CPT]
location.loc[21, "CPT"] = "CPT_05a"
del tmp


bathyFilesDirec = "cpt_raw_data/"
bathyFiles = glob.glob(bathyFilesDirec + "*.csv")
bathyFiles = np.sort([x.replace("datasets/cpt_raw_data/", "") for x in files])
train = np.array(
    [
        files[0],
        files[1],
        files[6],
        files[7],
        files[9],
        files[10],
        files[11],
        files[13],
        files[14],
        files[16],
        files[17],
        files[18],
        files[19],
        files[20],
        files[22],
    ]
)
test = np.array([files[8], files[12], files[15], files[21]])

cols = ["Depth", "Smooth qt", "Smooth fs", "longitude", "latitude"]
train_df = pd.DataFrame()
for f in train:  #
    df = pd.read_csv(
        direc + f,
        encoding="unicode_escape",
        skip_blank_lines=True,
        usecols=cols,
    )
    p_data = pd.read_csv(
        bathyFilesDirec + f,
        encoding="unicode_escape",
        nrows=6,
        header=None,
        index_col=[0],
        usecols=[0, 1],
    )  # point data for lat/lng and depth
    df = df.dropna()

    df["bathymetry"] = np.ones(len(df)) * float(p_data.loc["Water Depth", 1])

    df["lat"] = np.ones(len(df)) * float(
        location[location["CPT"] == f[:-4]]["lat"]
    )

    df["lng"] = np.ones(len(df)) * float(
        location[location["CPT"] == f[:-4]]["lng"]
    )
    # df["u2"] = pd.read_csv(bathyFilesDirec + f, skiprows=8).dropna()[
    #     "Pore pressure u2"
    # ]
    train_df = pd.concat([train_df, df])
train_df = train_df.rename(
    columns={
        # "Cone Resistance qc": "qc",
        # "Corrected Cone Resistance qt": "qt",
        # "Sleeve Friction fs": "fs",
    }
)
train_df.to_csv("train.csv", index=False)


test_df = pd.DataFrame()
for f in test:
    if f != "CPT_11.csv":
        df = pd.read_csv(
            direc + f,
            encoding="unicode_escape",
            skip_blank_lines=True,
            usecols=cols,
        )
        p_data = pd.read_csv(
            bathyFilesDirec + f,
            encoding="unicode_escape",
            nrows=6,
            header=None,
            index_col=[0],
            usecols=[0, 1],
        )  # point data for lat/lng and depth
        df = df.dropna()

        df["bathymetry"] = np.ones(len(df)) * float(
            p_data.loc["Water Depth", 1]
        )

        df["lat"] = np.ones(len(df)) * float(
            location[location["CPT"] == f[:-4]]["lat"]
        )

        df["lng"] = np.ones(len(df)) * float(
            location[location["CPT"] == f[:-4]]["lng"]
        )
        df["u2"] = pd.read_csv(bathyFilesDirec + f, skiprows=8).dropna()[
            "Pore pressure u2"
        ]
        test_df = pd.concat([test_df, df])

f = "CPT_11.csv"
df = pd.read_csv(
    direc + f, encoding="unicode_escape", skip_blank_lines=True, usecols=cols
)
p_data = pd.read_csv(
    bathyFilesDirec + f,
    encoding="unicode_escape",
    nrows=6,
    header=None,
    index_col=[0],
    usecols=[0, 1],
)  # point data for lat/lng and depth
df = df.dropna()

df["bathymetry"] = np.ones(len(df)) * float(46.957413)
df["lat"] = np.ones(len(df)) * float(
    location[location["CPT"] == f[:-4]]["lat"]
)

df["lng"] = np.ones(len(df)) * float(
    location[location["CPT"] == f[:-4]]["lng"]
)
test_df = pd.concat([test_df, df])

test_df = test_df.rename(
    columns={
        "Cone Resistance qc": "qc",
        "Corrected Cone Resistance qt": "qt",
        "Sleeve Friction fs": "fs",
    }
)

test_df.to_csv("test.csv", index=False)
