import argparse
import importlib
import getpass
from glob import glob
import os
import shutil
import sys
from time import time

import numpy as np
from skimage.io import imread

import matplotlib
import matplotlib.pyplot as plt
LM_COLORS = matplotlib.cm.jet(np.linspace(0, 1, 68))

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='e.g. /home/tomas/brighterai/PRNet/predict_prnet_tf.py')
parser.add_argument('--predictor', default='predict', help='Function in predictor file that predicts uv posmap from (image). Default: predict')
parser.add_argument('--save_results', default='', help='Name to save the results as')
parser.add_argument('--debug', '-d', action='store_true', help='print debug info')
opt = parser.parse_args()

uv_kpt_ind = np.loadtxt('Data/uv-data/uv_kpt_ind.txt').astype(np.int32)
uv_mask = imread('Data/uv-data/uv_face_weight_mask.png') / 255.
uv_mask = np.array([uv_mask, uv_mask, uv_mask]).T

base_folder = '/mnt/candy/datasets/AFLW2000-3D/AFLW2000-3D-UV-posmaps'
base_path_list = glob(os.path.join(base_folder, 'image*'))
total_num = len(base_path_list)
print('Running evaluation script on {} images in total'.format(total_num))

sys.path.append(opt.path)
preditor_module = importlib.import_module(opt.path.split('/')[-1].split('.')[0])
Predictor = getattr(preditor_module, opt.predictor)


def get_bbox_from_landmarks(kpt):
    if kpt.shape[0] > 3:
        kpt = kpt.T
    left = np.min(kpt[0, :])
    right = np.max(kpt[0, :])
    top = np.min(kpt[1, :])
    bottom = np.max(kpt[1, :])
    return np.array([left, right, top, bottom], dtype=np.int32)


def get_landmarks_from_uvmap(uvmap):
    kpt = uvmap[uv_kpt_ind[1, :], uv_kpt_ind[0, :], :]
    # kpt[:, 2] = kpt[:, 2] - np.mean(kpt[:, 2])
    kpt = kpt.T
    return kpt


def predict_3D_landmarks(image):

    image = image / 255.

    stime = time()
    uvmap = Predictor(image)
    ntime = time() - stime

    kpt = get_landmarks_from_uvmap(uvmap)

    return uvmap, kpt, ntime


def draw_lms(path, image, real_lm, tf_lm, real_uv, tf_uv):
    if opt.debug:
        print('Saving plotted landmarks into {}'.format(path))

    plt.figure(figsize=(10, 10))

    if opt.debug:
        print('real_lm {} {} {} {} {}'.format(real_lm.dtype, real_lm.mean(), real_lm.shape, np.max(real_lm), np.min(real_lm)))
        print('tf_lm {} {} {} {} {}'.format(tf_lm.dtype, tf_lm.mean(), tf_lm.shape, np.max(tf_lm), np.min(tf_lm)))
    ax = plt.subplot(221)
    ax.set_title("Original")
    plt.imshow(image)
    plt.scatter(real_lm[0, :], real_lm[1, :], s=8, c=LM_COLORS)
    ax = plt.subplot(222)
    ax.set_title("TensoFlow")
    plt.imshow(image)
    plt.scatter(tf_lm[0, :], tf_lm[1, :], s=8, c=LM_COLORS)

    if opt.debug:
        print('real_uv {} {} {} {} {}'.format(real_uv.dtype, real_uv.mean(), real_uv.shape, np.max(real_uv), np.min(real_uv)))
        print('tf_uv {} {} {} {} {}'.format(tf_uv.dtype, tf_uv.mean(), tf_uv.shape, np.max(tf_uv), np.min(tf_uv)))
    plt.subplot(223)
    plt.imshow(real_uv / 255)
    plt.subplot(224)
    plt.imshow(tf_uv / 255)

    plt.savefig(path)


N = 0
NME_2D = 0
NME_3D = 0
MSE_UV = 0
prediction_time = 0
for i, base_path in enumerate(base_path_list):

    if opt.debug:
        print('##############################################################################################')

    image_name = base_path.split('/')[-1]
    image_path = base_path + '/' + image_name + '.jpg'
    image = imread(image_path)

    save_base_path = image_path.replace('AFLW2000-3D-UV-posmaps', 'AFLW2000-3D-UV-posmaps-' + opt.save_results).split('.')[0]

    uv_path = base_path + '/' + base_path.split('/')[-1] + '.npy'
    real_uv = np.load(uv_path)
    real_landmarks = get_landmarks_from_uvmap(real_uv)

    bbox = get_bbox_from_landmarks(real_landmarks)
    bbox_size = ((bbox[1] - bbox[0]) + (bbox[3] - bbox[2])) / 2

    # pred_uv, pred_landmarks = predict_3D_landmarks(image, np.array([bbox[0], bbox[1], bbox[2], bbox[3]]))
    pred_uv, pred_landmarks, ntime = predict_3D_landmarks(image)

    if opt.debug:
        print('uv_mask shape {}'.format(uv_mask.shape))
        print('uv_mask mean {} max {} min {}'.format(np.mean(uv_mask), np.max(uv_mask), np.min(uv_mask)))
        mean_uv = np.mean(real_uv, axis=(0, 1))
        min_uv = np.min(real_uv, axis=(0, 1))
        max_uv = np.max(real_uv, axis=(0, 1))
        print('real_uv shape {}'.format(real_uv.shape))
        print('real_uv mean: [{:.2f}, {:.2f}, {:.2f}]'.format(mean_uv[0], mean_uv[1], mean_uv[2]))
        print('real_uv min: [{:.2f}, {:.2f}, {:.2f}]'.format(min_uv[0], min_uv[1], min_uv[2]))
        print('real_uv max: [{:.2f}, {:.2f}, {:.2f}]'.format(max_uv[0], max_uv[1], max_uv[2]))
        mean_uv = np.mean(pred_uv, axis=(0, 1))
        min_uv = np.min(pred_uv, axis=(0, 1))
        max_uv = np.max(pred_uv, axis=(0, 1))
        print('pred_uv shape {}'.format(pred_uv.shape))
        print('pred_uv mean: [{:.2f}, {:.2f}, {:.2f}]'.format(mean_uv[0], mean_uv[1], mean_uv[2]))
        print('pred_uv min: [{:.2f}, {:.2f}, {:.2f}]'.format(min_uv[0], min_uv[1], min_uv[2]))
        print('pred_uv max: [{:.2f}, {:.2f}, {:.2f}]'.format(max_uv[0], max_uv[1], max_uv[2]))
        print('real_landmarks shape {}'.format(real_landmarks.shape))
        print('real_landmarks means {}'.format(np.mean(real_landmarks, axis=1)))
        print('pred_landmarks shape {}'.format(pred_landmarks.shape))
        print('pred_landmarks means {}'.format(np.mean(pred_landmarks, axis=1)))

    if i <= 10 and opt.save_results:
        visualisations_path = '/home/{}/test_prnet_{}'.format(getpass.getuser(), opt.save_results)
        os.makedirs(visualisations_path, exist_ok=True)
        draw_lms(os.path.join(visualisations_path, image_path.split('/')[-1]),
                 image, real_landmarks, pred_landmarks, real_uv, pred_uv)

    if opt.save_results != '':
        os.makedirs('/'.join(save_base_path.split('/')[:-1]), exist_ok=True)
        shutil.copy(image_path, save_base_path + '.jpg')
        np.save(save_base_path + '.npy', pred_uv)

    error_2D = np.sum(np.sqrt((real_landmarks[:2, :] - pred_landmarks[:2, :]) ** 2))
    error_2D = (error_2D / 68.) / bbox_size  # per landmark, normalize by bbox size
    if pred_landmarks.shape[0] == 3:
        error_3D = np.sum(np.sqrt((real_landmarks[:, :] - pred_landmarks[:, :]) ** 2))
        error_3D = (error_3D / 68.) / bbox_size  # per landmark, normalize by bbox size
    else:
        error_3D = -1
        NME_3D = -1
    if pred_uv is not None:
        error_uv = np.mean((uv_mask * (real_uv - pred_uv)) ** 2)
    else:
        error_uv = -1
        MSE_UV = -1

    prediction_time = ((prediction_time * N) + ntime) / (N + 1)
    NME_2D = ((NME_2D * N) + error_2D) / (N + 1)
    NME_3D = ((NME_3D * N) + error_3D) / (N + 1)
    MSE_UV = ((MSE_UV * N) + error_uv) / (N + 1)
    N += 1

    print(
        '{: >4d} / {: >4d} = {:.2%}  |  NME_2D: {:.2%}  |  NME_3D: {:.2%}  |  MSE_UV: {:.2f}  |  time to predict 1: {:.3f}s'.format(
        i,
        len(base_path_list),
        i / len(base_path_list),
        NME_2D, NME_3D, MSE_UV, prediction_time))

print('######\n final results  |  N:{}  |  NME_2D: {:.2%}  |  NME_3D: {:.2%}  |  MSE_UV: {:.2f}  |  time to predict 1: {:.3f}s \n######'.format(
    N, NME_2D, NME_3D, MSE_UV, prediction_time))

