import cv2
import numpy as np


def predict(pd_input, img_size, model):
    pd_input = pd_input[..., ::-1]
    pd_input = cv2.resize(pd_input, img_size)
    pd_input = np.reshape(pd_input, [1, img_size[0], img_size[1], 3]) * 1.0
    pd_input /= 127.5
    pd_input -= 1.0
    output = model.predict(pd_input)[0]
    return output
