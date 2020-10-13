import numpy as np

file_path = "./20200926_6.txt"
time_part = 12
kitti_frame = 0
nusc_frame = 0
time_frame = np.zeros(time_part)
intact_flag = True
kitti_time_all = np.zeros((2,time_part))
nusc_time_all = np.zeros((2,time_part))

with open(file_path) as f:
    for line in f:
        if line.find('start to') != -1:
            time_frame[0] = float(line.replace('start to min_points_in_gt:','').strip())
        if line.find('min_points_in_gt duration') != -1:
            time_frame[1] = float(line.replace('min_points_in_gt duration time:','').strip())
        if line.find('flip') != -1:
            time_frame[2] = float(line.replace('flip duration time:','').strip())
        if line.find('gt sample duration') != -1 and line.find('downsample') == -1:
            time_frame[3] = float(line.replace('gt sample duration time:','').strip())
        if line.find('noiseobject') != -1:
            time_frame[4] = float(line.replace('noiseobject duration time:','').strip())
        if line.find('rot&scal') != -1:
            time_frame[5] = float(line.replace('rot&scal duration time:','').strip())
        if line.find('front range filter') != -1:
            time_frame[6] = float(line.replace('front range filter duration time:','').strip())
        if line.find('downsample') != -1:
            time_frame[7] = float(line.replace('downsample duration time:','').strip())
        if line.find('filter_outrange') != -1:
            time_frame[8] = float(line.replace('filter_outrange duration time: ','').strip())
        if line.find('shuffle') != -1:
            time_frame[9] = float(line.replace('shuffle duration time:','').strip())
        if line.find('random_select') != -1:
            time_frame[10] = float(line.replace('random_select duration time: ','').strip())
        if line.find('data preprocess') != -1:
            time_frame[11] = float(line.replace('data preprocess time:','').strip())

        if line.find('/home/dataset/') != -1:
            print(time_frame)
            for i in range(time_part):
                if time_frame[i]==0:
                    print(time_frame[i])
                    intact_flag = False
                    break

            if intact_flag:
                if line.find('KITTI') != -1:
                    if kitti_frame == 0:
                        kitti_time_all = time_frame
                    else:
                        kitti_time_all = np.vstack((kitti_time_all,time_frame))
                    kitti_frame += 1
                    print("kitti_:",kitti_frame)
                else:
                    if nusc_frame == 0:
                        nusc_time_all = time_frame
                    else:
                        nusc_time_all = np.vstack((nusc_time_all,time_frame))
                    nusc_frame += 1
                    print("nusc_:",nusc_frame)
            intact_flag = True
            time_frame = np.zeros(time_part)
print("nusc:")
for i in range(time_part):
    print(i,":",np.mean(nusc_time_all[10:-10,i])*1000)

print("kitti:")
for i in range(time_part):
    print(i,":",np.mean(kitti_time_all[10:-10,i])*1000)

