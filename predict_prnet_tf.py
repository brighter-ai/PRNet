import os

from api import PRN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

prn = PRN(is_dlib=False)


def predict(image):
    return prn.net_forward(image)
