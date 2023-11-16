import array
import OpenEXR
import Imath
import numpy as np
from tqdm import trange


def load_exr(path):
    exr = OpenEXR.InputFile(path)

    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    img_R, img_G, img_B = exr.channels('RGB', pt)

    arr_R = array.array('f', img_R)
    arr_G = array.array('f', img_G)
    arr_B = array.array('f', img_B)

    ndarr_R = np.array(arr_R, dtype="float32")
    ndarr_G = np.array(arr_G, dtype="float32")
    ndarr_B = np.array(arr_B, dtype="float32")

    ndarr_RGB = np.stack([ndarr_R, ndarr_G, ndarr_B], axis=1)

    ndarr_RGB = ndarr_RGB.reshape(size[1], size[0], 3)

    return ndarr_RGB


def preview_exr(path):
    img = load_exr(path)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(img)
    plt.show()


def convert_matrix(img, matrix):
    out = np.zeros(img.shape)
    for _i in trange(img.shape[0]):
        for _j in range(img.shape[1]):
            out[_i, _j, :] = np.dot(matrix, img[_i, _j, :])
    return out


def method_okawa(input, output):
    output /= output.max()
    return np.linalg.inv(np.dot(output, np.linalg.pinv(input)))


def method_ue(input, output):
    W = output[:, 3]
    output = output[:, :3]
    Imat = np.linalg.inv(output)
    Wnor = W / W.max()
    S = np.dot(Imat, Wnor)
    Smat = np.diag(S)
    C = np.dot(Smat, output)
    return np.linalg.inv(C)
