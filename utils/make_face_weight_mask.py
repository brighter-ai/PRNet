import numpy as np
from skimage.io import imread, imsave

weight_mask = imread('../Data/uv-data/uv_weight_mask.png')
print('  |  '.join([
        'weight_mask:',
        'shape: {}'.format(weight_mask.shape),
        'type: {}'.format(weight_mask.dtype),
        'min: {}'.format(np.min(weight_mask)),
        'max: {}'.format(np.max(weight_mask))
    ]))

face_mask = imread('../Data/uv-data/uv_face_mask.png')
face_mask = (face_mask / 255).astype(np.uint8)
print('  |  '.join([
        'face_mask:',
        'shape: {}'.format(face_mask.shape),
        'type: {}'.format(face_mask.dtype),
        'min: {}'.format(np.min(face_mask)),
        'max: {}'.format(np.max(face_mask))
    ]))

face_weight_mask = weight_mask * face_mask
print('  |  '.join([
        'face_weight_mask:',
        'shape: {}'.format(face_weight_mask.shape),
        'type: {}'.format(face_weight_mask.dtype),
        'min: {}'.format(np.min(face_weight_mask)),
        'max: {}'.format(np.max(face_weight_mask))
    ]))

imsave('../Data/uv-data/uv_face_weight_mask.png', face_weight_mask)