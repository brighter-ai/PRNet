import argparse
import importlib
from glob import glob
import os
import sys

import numpy as np
import scipy.io as sio
from skimage.io import imread


parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='e.g. /home/tomas/brighterai/PRNet/predict_3D_landmarks.py')
parser.add_argument('--predictor', default='predict_lms', help='Function in predictor file that predicts 3D lms (shape (3,68)) from (image, bbox). Default: predict_lms')
opt = parser.parse_args()

image_folder = '/mnt/candy/datasets/AFLW2000-3D/AFLW2000-3D'
image_path_list = glob(os.path.join(image_folder, '*.jpg'))
total_num = len(image_path_list)
print('Running evaluation script on {} images in total'.format(total_num))

sys.path.append(opt.path)
preditor_module = importlib.import_module(opt.path.split('/')[-1].split('.')[0])
predict_3D_landmarks = getattr(preditor_module, opt.predictor)


def get_bbox_from_landmarks(kpt):
    if kpt.shape[0] > 3:
        kpt = kpt.T
    left = np.min(kpt[0, :])
    right = np.max(kpt[0, :])
    top = np.min(kpt[1, :])
    bottom = np.max(kpt[1, :])
    return np.array([left, right, top, bottom], dtype=np.int32)


N = 0
NME_2D = 0
NME_3D = 0
for i, image_path in enumerate(image_path_list):

    image = imread(image_path)

    mat_path = image_path.replace('jpg', 'mat')
    info = sio.loadmat(mat_path)
    real_landmarks = info['pt3d_68']
    bbox = get_bbox_from_landmarks(real_landmarks)
    bbox_size = ((bbox[1] - bbox[0]) + (bbox[3] - bbox[2])) / 2

    pred_landmarks = predict_3D_landmarks(image, bbox)

    error_2D = np.sum(np.sqrt((real_landmarks[:2, :] - pred_landmarks[:2, :]) ** 2))
    error_2D = (error_2D / 68.) / bbox_size  # per landmark, normalize by bbox size
    error_3D = np.sum(np.sqrt((real_landmarks[:, :] - pred_landmarks[:, :]) ** 2))
    error_3D = (error_3D / 68.) / bbox_size  # per landmark, normalize by bbox size

    NME_2D = ((NME_2D * N) + error_2D) / (N + 1)
    NME_3D = ((NME_3D * N) + error_3D) / (N + 1)
    N += 1

    print('{: >4d} / {: >4d} = {:.2%}  |   NME_2D (so far): {:.2%}  |   NME_3D (so far): {:.2%}'.format(
        i,
        len(image_path_list),
        i / len(image_path_list),
        NME_2D,
        NME_3D))

print('######\n final results  |  N:{}  |  NME_2D: {:.2%}  |  NME_3D: {:.2%} \n######'.format(N, NME_2D, NME_3D))

