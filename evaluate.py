import argparse
import importlib
from glob import glob
import os
import sys

import numpy as np
import scipy.io as sio
from skimage.io import imread
from skimage.transform import estimate_transform, warp


parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='e.g. /home/tomas/brighterai/PRNet/predict_prnet_tf.py')
parser.add_argument('--predictor', default='predict', help='Function in predictor file that predicts uv posmap from (image). Default: predict')
parser.add_argument('--debug', '-d', action='store_true', help='print debug info')
opt = parser.parse_args()

uv_kpt_ind = np.loadtxt('/home/tomas/brighterai/3dnet/assets/uv_kpt_ind.txt').astype(np.int32)
image_folder = '/mnt/candy/datasets/AFLW2000-3D/AFLW2000-3D'
image_path_list = glob(os.path.join(image_folder, '*.jpg'))
total_num = len(image_path_list)
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


def predict_3D_landmarks(image, bbox):

    # Crop the image based on bbox of the face
    resolution = 256
    left = bbox[0]; right = bbox[1]; top = bbox[2]; bottom = bbox[3]
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * 1.6)
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, resolution - 1], [resolution - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    image = image / 255.
    cropped_image = warp(image, tform.inverse, output_shape=(resolution, resolution))

    # Predict uvmap
    cropped_uvmap = Predictor(cropped_image)

    # Convert predicted uvmap back to original image dimensions
    cropped_vertices = np.reshape(cropped_uvmap, [-1, 3]).T
    z = cropped_vertices[2, :].copy() / tform.params[0, 0]
    cropped_vertices[2, :] = 1
    vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
    vertices = np.vstack((vertices[:2, :], z))
    uvmap = np.reshape(vertices.T, [resolution, resolution, 3])

    # Get landmarks from uvmap
    kpt = uvmap[uv_kpt_ind[1, :], uv_kpt_ind[0, :], :]
    kpt[:, 2] = kpt[:, 2] - np.mean(kpt[:, 2])
    kpt = kpt.T

    return uvmap, kpt


N = 0
NME_2D = 0
NME_3D = 0
MSE_UV = 0
for i, image_path in enumerate(image_path_list[:10]):

    image = imread(image_path)

    mat_path = image_path.replace('.jpg', '.mat')
    info = sio.loadmat(mat_path)
    real_landmarks = info['pt3d_68']
    uv_path = image_path.replace('AFLW2000-3D/AFLW2000-3D', 'AFLW2000-3D/AFLW2000-3D-UV-posmaps')
    uv_path += '/' + uv_path.split('/')[-1].replace('.jpg', '.npy')
    real_uv = np.load(uv_path)
    uv_mask = imread('/home/tomas/brighterai/3dnet/assets/uv_weight_mask.png') / 255.
    uv_mask = np.array([uv_mask, uv_mask, uv_mask]).T
    bbox = get_bbox_from_landmarks(real_landmarks)
    bbox_size = ((bbox[1] - bbox[0]) + (bbox[3] - bbox[2])) / 2

    # pred_uv, pred_landmarks = predict_3D_landmarks(image, np.array([bbox[0], bbox[1], bbox[2], bbox[3]]))
    pred_uv, pred_landmarks = predict_3D_landmarks(image, bbox)

    if opt.debug:
        print('uv_mask shape {}'.format(uv_mask.shape))
        print('uv_mask mean {} max {} min {}'.format(np.mean(uv_mask), np.max(uv_mask), np.min(uv_mask)))
        print('real_uv shape {}'.format(real_uv.shape))
        print('real_uv means {}'.format(np.mean(real_uv, axis=(0, 1))))
        print('pred_uv shape {}'.format(pred_uv.shape))
        print('pred_uv means {}'.format(np.mean(pred_uv, axis=(0, 1))))
        print('real_landmarks shape {}'.format(real_landmarks.shape))
        print('real_landmarks means {}'.format(np.mean(real_landmarks, axis=1)))
        print('pred_landmarks shape {}'.format(pred_landmarks.shape))
        print('pred_landmarks means {}'.format(np.mean(pred_landmarks, axis=1)))

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

    NME_2D = ((NME_2D * N) + error_2D) / (N + 1)
    NME_3D = ((NME_3D * N) + error_3D) / (N + 1)
    MSE_UV = ((MSE_UV * N) + error_uv) / (N + 1)
    N += 1

    print('{: >4d} / {: >4d} = {:.2%}  |  NME_2D (so far): {:.2%}  |  NME_3D (so far): {:.2%}  |  MSE_UV (so far): {:.2f}'.format(
        i,
        len(image_path_list),
        i / len(image_path_list),
        NME_2D, NME_3D, MSE_UV))

print('######\n final results  |  N:{}  |  NME_2D: {:.2%}  |  NME_3D: {:.2%}  |  MSE_UV: {:.2f} \n######'.format(
    N, NME_2D, NME_3D, MSE_UV))

