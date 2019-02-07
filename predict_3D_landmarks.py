import os

import numpy as np

from api import PRN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

prn = PRN(is_dlib=False)


def predict_lms(image, bbox):
    pos = prn.process(image, bbox)
    pred_landmarks = prn.get_landmarks(pos)
    pred_landmarks[:, 2] = pred_landmarks[:, 2] - np.mean(pred_landmarks[:, 2])
    pred_landmarks = pred_landmarks.T
    return pred_landmarks
