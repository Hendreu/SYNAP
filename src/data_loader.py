import numpy as np
import pandas as pd

def load_data(csv_path, dev_size=1000):
    data = pd.read_csv(csv_path)
    data = np.array(data)
    m, n = data.shape

    np.random.shuffle(data)

    data_dev = data[:dev_size].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]

    data_train = data[dev_size:].T
    Y_train = data_train[0]
    X_train = data_train[1:n]

    X_train = X_train / 255.0
    X_dev = X_dev / 255.0

    return X_train, Y_train, X_dev, Y_dev
