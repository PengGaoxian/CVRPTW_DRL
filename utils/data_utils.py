import os
import pickle


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        f.flush() # 清空缓冲区
        os.fsync(f.fileno()) # 确保数据被写入磁盘


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)