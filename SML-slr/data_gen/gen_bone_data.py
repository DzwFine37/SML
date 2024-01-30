import os
import numpy as np
from numpy.lib.format import open_memmap

paris = {
    'slr-27': ((5, 6), (5, 7), (6, 8), (8, 10), (7, 9), (9, 11), (12,13),(12,14),(12,16),(12,18),(12,20),
(14,15),(16,17),(18,19),(20,21), (22,23),(22,24),(22,26),(22,28),(22,30),(24,25),(26,27),(28,29),(30,31),
(10,12),(11,22)
    )
}

sets = {
    'train', 'val', 'test'
}

datasets = {
    'slr-27'
}

from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            '../data/{}/{}_data_bone.npy'.format(dataset, set),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

        fp_sp[:, :C, :, :, :] = data
        for v1, v2 in tqdm(paris[dataset]):
            v1 -= 5
            v2 -= 5
            fp_sp[:, :, :, v2, :] = data[:, :, :, v2, :] - data[:, :, :, v1, :]

