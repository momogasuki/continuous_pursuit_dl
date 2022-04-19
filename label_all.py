from util import *
import os, pickle

if __name__ == '__main__':
    data = {}
    for root, dirs, files in os.walk('/data/datasets/sciRobCP/data/'):
        for file in files:
            if file[-3:] == 'mat':
                if 'LR' in file: continue
                if 'UD' in file: continue
                if 'DT' in file: continue
                path = os.path.join(root, file)
                subj = root.split('/')[-1]
                run = file[:-4]
                data.setdefault(subj, {})[run] = get_data(path)
    with open('tmp.pkl', 'wb') as f:
        pickle.dump(data, f)