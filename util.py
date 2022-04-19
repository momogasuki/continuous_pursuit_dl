import mat73
import scipy.io as scio
import numpy as np

# path = "/data/datasets/sciRobCP/ESI_Exp_S13/S13_CP_2D_ESI03_R01.mat"

def get_data(path):
    try:
        data_dict = mat73.loadmat(path)
    except:
        data_dict = scio.loadmat(path, simplify_cells=True)

    tar_x = data_dict['eeg']['targetpos']['x']
    tar_y = data_dict['eeg']['targetpos']['y']
    cur_x = data_dict['eeg']['cursorpos']['x']
    cur_y = data_dict['eeg']['cursorpos']['y']
    eeg = data_dict['eeg']['data']
    fs = data_dict['eeg']['fs']

    ## deal with edge case
    tar_x[1:] -= np.diff(tar_x).round().cumsum()
    tar_y[1:] -= np.diff(tar_y).round().cumsum()
    cur_x[1:] -= np.diff(cur_x).round().cumsum()
    cur_y[1:] -= np.diff(cur_y).round().cumsum()

    window_length = 0.5 # in seconds
    window_stride = 0.039
    # class_num = 4

    X, Y = [], []

    for ev in data_dict['eeg']['event']:
        if ev['type'] == 'TrialStart':
            ev_op = ev['latency']/1000
            ev_ed = (ev['latency']+ev['duration'])/1000 - window_length
            window_start_t = ev_op
            while window_start_t <= ev_ed:
                op = int(window_start_t*fs)
                l = int(window_length*fs)
                ed = op + l
                if ed > eeg.shape[1]: break
                pos_idx = (data_dict['eeg']['postimes'] >= window_start_t) & (data_dict['eeg']['postimes'] < window_start_t + window_length)
                dir_x = tar_x[pos_idx].mean() - cur_x[pos_idx].mean()
                dir_y = tar_y[pos_idx].mean() - cur_y[pos_idx].mean()
                dir_x -= dir_x.round()
                dir_y -= dir_y.round()
                dir_ori_pi = np.arctan2(dir_y, dir_x) / np.pi # (-1,1]
                label = [round((dir_ori_pi)*class_num/2)%class_num for class_num in [4,8,16,32]]
                label.append(dir_ori_pi)
                X.append(eeg[:,op:ed])
                Y.append(label)

                window_start_t += window_stride

    X = np.array(X)
    Y = np.array(Y)

    print('Shape of data is', X.shape, '(trial*channel*timestamp)')
    print('Shape of label is', Y.shape)
    return X, Y