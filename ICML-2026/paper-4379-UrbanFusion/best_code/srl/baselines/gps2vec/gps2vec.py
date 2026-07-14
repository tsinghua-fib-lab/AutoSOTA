"""
Adapted code from: https://github.com/yifangyin/GPS2Vec
"""

import math
import os
import sys

import numpy as np
import utm

os.environ["KERAS_BACKEND"] = "torch"
from keras.models import model_from_json

# Force Keras 3 to use the PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

# Import Keras 3
import keras
from keras.layers import Dense, Flatten, Input
from keras.models import Model


# Define builder for the “visual” model (flag=0)
def build_visual_model(n_rows=20, n_cols=20):
    inp = Input(shape=(n_rows, n_cols, 1), name="input_1")
    x = Flatten(name="flatten_1")(inp)
    x = Dense(512, activation="relu", name="layer1")(x)
    x = Dense(1024, activation="relu", name="layer2")(x)
    x = Dense(2048, activation="relu", name="layer3")(x)
    out_object = Dense(1000, activation="softmax", name="object")(x)
    out_scene = Dense(365, activation="softmax", name="scene")(x)
    model = Model(inputs=inp, outputs=[out_object, out_scene], name="model_1")
    return model


# Define builder for the “tag” model (flag=1)
def build_tag_model(n_rows=20, n_cols=20):
    inp = Input(shape=(n_rows, n_cols, 1), name="input_1")
    x = Flatten(name="flatten_1")(inp)
    x = Dense(512, activation="relu", name="layer1")(x)
    x = Dense(1024, activation="relu", name="layer2")(x)
    x = Dense(2048, activation="relu", name="layer3")(x)
    out_tag = Dense(2000, activation="softmax", name="tag_output")(x)
    model = Model(inputs=inp, outputs=out_tag, name="model_1")
    return model


def load_model(basedir, utmzone, nrows, ncols, flag):
    """
    Rebuild the architecture in Keras 3 (torch) and load weights from .h5.
      - flag == 0 → visual model (two‐head outputs).
      - flag == 1 → tag model (single‐head).
    """
    if flag == 0:
        model = build_visual_model(nrows, ncols)
        h5_path = f"{basedir}/model_{utmzone}.h5"
    else:
        model = build_tag_model(nrows, ncols)
        h5_path = f"{basedir}/model_{utmzone}.h5"

    # Load the weights. As long as layer names match exactly, this works.
    model.load_weights(h5_path)
    return model


def utm_zone_ranges():
    # apart from a few exceptions around Norway and Svalbard: 31V,32V,31X,33X,35X,37X
    # https://github.com/Turbo87/utm/blob/master/utm/conversion.py
    epsilon = 0
    utm_ranges = {}
    lat_bands = [
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "J",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
    ]
    # longitude from west to east, 6 degrees each
    for i in range(60):
        lon_band = str(i + 1)
        lon_min = -180 + i * 6
        lon_max = -180 + (i + 1) * 6 - epsilon
        # latitude from south to north (80 S to 72 N to 84 N), 8 degrees each
        for j in range(20):
            lat_band = lat_bands[j]
            utmzone = lon_band + " " + lat_band
            lat_min = -80 + j * 8
            lat_max = -80 + (j + 1) * 8 - epsilon
            if j == 19:
                lat_max = lat_max + 4

            if (
                utmzone == "31 V"
                or utmzone == "32 X"
                or utmzone == "34 X"
                or utmzone == "36 X"
            ):
                continue
            # deal with exceptions: 32V, 31X, 33X, 35X, 37X
            if utmzone == "32 V":
                lon_min = 3
                lon_max = 12
            elif utmzone == "31 X":
                lon_min = 0
                lon_max = 9
            elif utmzone == "33 X":
                lon_min = 9
                lon_max = 21
            elif utmzone == "35 X":
                lon_min = 21
                lon_max = 33
            elif utmzone == "37 X":
                lon_min = 33
                lon_max = 42
            else:
                lon_min = -180 + i * 6
                lon_max = -180 + (i + 1) * 6 - epsilon

            utm_ranges[utmzone] = np.asarray(
                [lon_min, lon_max, lat_min, lat_max]
            )

    return utm_ranges


def loc2mat(location, nrows=20, ncols=20, sigma=20000):
    """
    utm format: (EASTING, NORTHING, ZONE NUMBER, ZONE LETTER)
    Northing -> row
    Easting -> col

    http://www.dmap.co.uk/utmworld.htm
    For the eastings, the origin is defined as a point 500,000 metres west of the central meridian of each longitudinal zone, giving an easting of 500,000 metre at the central meridian. eastings are usually less than 834 000 m, and more than 160 000 m.

    For the northings in the northern hemisphere, the origin is defined as the equator.
    For the northings in the southern hemisphere, the origin is defined as a point 10,000,000 metres south of the equator.
    The maximum "northing" value is about 9300000 meters at latitude 84 degrees North
    """
    latitude = location[0]
    longitude = location[1]
    utm_tuple = utm.from_latlon(latitude, longitude)
    lon_band = utm_tuple[2]
    lat_band = utm_tuple[3]
    utm_ranges = utm_zone_ranges()[str(lon_band) + " " + lat_band]
    lon_min = utm_ranges[0]
    lon_max = utm_ranges[1]
    lat_min = utm_ranges[2]
    lat_max = utm_ranges[3]

    epsilon = 1e-8
    bottomleft = utm.from_latlon(lat_min, lon_min)
    upleft = utm.from_latlon(lat_max - epsilon, lon_min)

    row_min = bottomleft[1]
    row_max = upleft[1]
    step = 0.1
    lon = lon_min

    while lon < lon_max:
        bottom = utm.from_latlon(lat_min, lon)
        up = utm.from_latlon(lat_max - epsilon, lon)
        if bottom[1] < row_min:
            row_min = bottom[1]
        if up[1] > row_max:
            row_max = up[1]
        lon += step

    col_min = min(bottomleft[0], upleft[0])
    col_max = 500000 + (500000 - col_min)

    mat = np.zeros([ncols, nrows])
    row_step = (row_max - row_min) / nrows
    col_step = (col_max - col_min) / ncols
    y = math.floor((utm_tuple[1] - row_min) / row_step)
    x = math.floor((utm_tuple[0] - col_min) / col_step)

    x = int(x)
    y = int(y)
    if x < 0:
        x = 0
        # print('warning: x<0')
    if y < 0:
        y = 0
        # print('warning: y<0')

    if x >= ncols:
        x = ncols - 1
    if y >= nrows:
        y = nrows - 1

    for xx in range(0, ncols):
        x_center = col_min + col_step * (xx + 0.5)
        for yy in range(0, nrows):
            y_center = row_min + row_step * (yy + 0.5)
            dist = np.linalg.norm(
                np.asarray([utm_tuple[0] - x_center, utm_tuple[1] - y_center])
            )
            mat[xx, yy] = np.exp(-dist / sigma)

    return mat


def georep(location, basedir, nrows, ncols, sigma, flag):
    if flag == 0:
        dim = 1000 + 365
    elif flag == 1:
        dim = 2000

    latitude = location[0]
    longitude = location[1]
    utmloc = utm.from_latlon(latitude, longitude)
    utmzone = str(utmloc[2]) + " " + utmloc[3]
    model_weights_file = f"{basedir}/model_{utmzone}.h5"
    if not os.path.exists(model_weights_file):
        print(f"Model weights file not found: {model_weights_file}")
        return np.full(dim, np.nan, dtype=np.float32)
    import h5py

    f = h5py.File(model_weights_file, "r")
    model = load_model(basedir, utmzone, nrows, ncols, flag)
    input = loc2mat(location, nrows, ncols, sigma)
    input = np.reshape(input, (1, input.shape[0], input.shape[1], 1))
    if flag == 0:
        output_fea = model.predict(input, verbose=0)
        geofeas = np.concatenate(
            (output_fea[0], output_fea[1]), axis=1
        ).flatten()
    elif flag == 1:
        output_fea = model.predict(input, verbose=0).flatten()
        geofeas = output_fea

    return geofeas


def main():
    nrows = 20
    ncols = 20
    sigma = 20000
    flag = 1
    if flag == 0:
        basedir = "./models_visual/"
    elif flag == 1:
        basedir = "./models_tag/"

    location = [1.3199909039789364, 103.764553967551]
    geofea = georep(location, basedir, nrows, ncols, sigma, flag)
    print(geofea)
    print(sum(geofea))


if __name__ == "__main__":
    main()
