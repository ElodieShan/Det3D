import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

def get_eval_result(file_path):
    eval_result = {}
    result_epoch = []
    flag = False
    with open(file_path) as f:
        for line in f:
            if flag and line.find('mean AP:')!=-1:
                line = line[:line.find('mean AP:')].split(',')
                eval_result[class_name].append(np.array([float(i.strip()) for i in line]))
                flag = False
                continue
            if line.find('Nusc dist AP')!= -1:
                class_name = line[:line.find('Nusc dist AP')].strip()
                if class_name not in eval_result:
                    eval_result[class_name] = []
                flag = True
    # eval_result = np.array(eval_result)
    plt.figure()
    for class_name, results in eval_result.items():
        results = np.array(results)
        x = np.arange(1,results.shape[0]+1)
        y = results[:,0]

        plt.plot(x,y,label=class_name, linewidth=0.5)
    plt.legend()
    plt.show()
    print("eval_result:",eval_result)
if __name__ == "__main__":
    log_file_root = "/mnt/DockerDet3D/det3D_Output/"
    log_file_dir = "NUSC_SECOND_9_20200923-175601/"
    log_file_name = "20200923_175624.log"
    log_file = log_file_root + log_file_dir + log_file_name
    get_eval_result(log_file)
