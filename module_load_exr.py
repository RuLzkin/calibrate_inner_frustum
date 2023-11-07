import array
import OpenEXR
import Imath
import numpy as np


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
