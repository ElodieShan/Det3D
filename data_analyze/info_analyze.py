import numpy as np
import pickle

def load_info(path):
    with open(path,"rb") as f:
        info = pickle.load(f)
    return info 

if __name__ == '__main__':
    root_path = "/home/dataset/KITTI_DATASET_NEW/object/pkl/"
    val_info_file = "kitti_infos_val_formatnusc_feature5.pkl"
    train_info_file = "kitti_infos_train_formatnusc_feature5_v3.pkl"
    val_info = load_info(root_path + val_info_file)
    train_info = load_info(root_path +train_info_file)
    # train_info2 = load_info(root_path +train_info_file)

    print("val_info:\n", val_info[0])
    print("train_info:\n", train_info[0])
