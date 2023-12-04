import numpy as np
import cv2
from matplotlib import pyplot as plt
from module_load_exr import load_exr


def lighting_dilation(img_input: np.ndarray, threshold_saturated: float = 1.0, str_kernel: str = "square", verbose: bool = False):
    """lighting_dilation

    img_input: np.ndarray
    threshold_saturated: float / default = 1.0
    str_kernel: "square" or "cross" / default = "square"
    verbose: bool / show graphics
    """

    # Dilation Kernel
    if str_kernel == "square":
        """ Square kernel """
        kernel = np.ones((3, 3))
    elif str_kernel == "cross":
        """ Cross kernel """
        kernel = np.zeros((3, 3), dtype=np.uint8)
        kernel[1, :], kernel[:, 1] = 1, 1

    img_output = img_input.copy()

    # (1)
    saturated_pixels = (img_output > threshold_saturated).astype(np.uint8)

    if verbose:
        fig, list_ax = plt.subplots(2, 3, figsize=(10, 6))
        fig.set_tight_layout(True)

    for _c, _color in enumerate(["RED", "GREEN", "BLUE"]):
        # (2)
        n_labels, labels = cv2.connectedComponents(saturated_pixels[..., _c])
        label_old = np.zeros_like(labels)
        label_old[:] = labels[:]

        # (3)
        flag_continue = True
        _n = 0
        if verbose:
            plt_img = list_ax[0, _c].imshow(img_input, vmin=0, vmax=1.0)
            plt_lbl = list_ax[1, _c].imshow((labels / n_labels) ** 0.1)
        while flag_continue:
            flag_continue = False
            _sum_dilated = 0
            for _i_label in range(1, n_labels):
                # (a)
                _x_bar = img_output[labels == _i_label, _c].mean()
                if np.isnan(_x_bar):
                    continue
                # (b)
                if threshold_saturated >= _x_bar:
                    img_output[labels == _i_label, _c] = _x_bar
                else:
                    _dilated = cv2.dilate((labels == _i_label).astype(np.uint8), kernel, iterations=1)
                    _dilated = np.logical_and(_dilated, np.logical_or(label_old == 0, labels == _i_label))
                    _sum_dilated += _dilated
                    labels[_dilated] = _i_label
                    # (c)
                    flag_continue = True
            labels[_sum_dilated > 1] = 0
            label_old[:] = labels[:]
            _n += 1
            if verbose:
                _tint = np.zeros(3)
                _tint[_c] = 1
                plt_img.set_data(np.maximum(0, np.minimum(1.0, img_output)) * _tint[None, None, :])
                plt_lbl.set_data((labels / n_labels) ** 0.1)
                list_ax[0, _c].set_title("iter: {0}, {1}".format(_n, _color))
                plt.pause(0.01)
            else:
                print("{0}\t({1}/{2}) iter: {3:5d}".format(_color, _c + 1, 3, _n), end="\r", flush=True)
        if verbose:
            print()
    return img_output


if __name__ == "__main__":

    img_input = load_exr(r"C:\Dropbox\TOEI\20231204_LightingDilation\test.exr")

    hdri_output = lighting_dilation(img_input, 1.0, verbose=True)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(img_input)
    plt.title("Input")
    plt.subplot(2, 1, 2)
    plt.imshow(hdri_output)
    plt.title("Output")
    plt.tight_layout()

    plt.show()
