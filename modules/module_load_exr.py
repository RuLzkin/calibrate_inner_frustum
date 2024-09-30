import array
import OpenEXR
import Imath
import numpy as np
from numpy import ndarray


def load_exr(path: str) -> ndarray:
    """load EXR file

        Args:
            path: str, path to EXR file

        Returns:
            np.ndarray, shape(row, column, [R, G, B])

        Examples:
            >>> nda_RGB = load_exr("path/to/file")
    """
    exr = OpenEXR.InputFile(path)

    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    img_R, img_G, img_B = exr.channels('RGB', pt)

    arr_R = array.array('f', img_R)
    arr_G = array.array('f', img_G)
    arr_B = array.array('f', img_B)

    ndarr_R: ndarray = np.array(arr_R, dtype="float32")
    ndarr_G: ndarray = np.array(arr_G, dtype="float32")
    ndarr_B: ndarray = np.array(arr_B, dtype="float32")

    ndarr_RGB = np.stack([ndarr_R, ndarr_G, ndarr_B], axis=1)

    ndarr_RGB = ndarr_RGB.reshape(size[1], size[0], 3)

    return ndarr_RGB


def preview_exr(path: str, show: bool = False, amp: float = 1.0) -> None:
    img = load_exr(path)
    from matplotlib import pyplot as plt
    plt.figure()
    if amp is None:
        plt.imshow(img / img.max())
    else:
        plt.imshow(amp * img)
    if show:
        plt.show()
