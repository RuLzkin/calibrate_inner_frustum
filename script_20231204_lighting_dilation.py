import numpy as np
import cv2
from matplotlib import pyplot as plt
from module_load_exr import load_exr

threshold_saturated = 1.0

""" Square kernel """
kernel = np.ones((3, 3))
""" Cross kernel """
# kernel = np.zeros((3, 3), dtype=np.uint8)
# kernel[1, :], kernel[:, 1] = 1, 1

hdri_input = load_exr(r"C:\Dropbox\TOEI\20231204_LightingDilation\test.exr")

hdri_output = hdri_input.copy()

# (1)
saturated_pixels = (hdri_output > threshold_saturated).astype(np.uint8)

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
    plt_img = list_ax[0, _c].imshow(hdri_input, vmin=0, vmax=1.0)
    plt_lbl = list_ax[1, _c].imshow((labels / n_labels) ** 0.1)
    while flag_continue:
        flag_continue = False
        _sum_dilated = 0
        for _i_label in range(1, n_labels):
            # (a)
            _x_bar = hdri_output[labels == _i_label, _c].mean()
            if np.isnan(_x_bar):
                continue
            # (b)
            if threshold_saturated >= _x_bar:
                hdri_output[labels == _i_label, _c] = _x_bar
            else:
                # print(threshold_saturated, _x_bar)
                _dilated = cv2.dilate((labels == _i_label).astype(np.uint8), kernel, iterations=1)
                _dilated = np.logical_and(_dilated, np.logical_or(label_old == 0, labels == _i_label))
                # _dilated = _dilated.astype(np.bool8)
                _sum_dilated += _dilated
                labels[_dilated] = _i_label
                flag_continue = True
                # print("xbar = {0}, i_label = {1}".format(_x_bar, _i_label))
        labels[_sum_dilated > 1] = 0
        label_old[:] = labels[:]
        _n += 1
        # print(_n, flush=True)
        _tint = np.zeros(3)
        _tint[_c] = 1
        plt_img.set_data(np.maximum(0, np.minimum(1.0, hdri_output)) * _tint[None, None, :])
        plt_lbl.set_data((labels / n_labels) ** 0.1)
        list_ax[0, _c].set_title("iter: {0}, {1}".format(_n, _color))
        # fig.tight_layout()
        plt.pause(0.01)


plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(hdri_input)
plt.title("Input")
plt.subplot(2, 1, 2)
plt.imshow(hdri_output)
plt.title("Output")
plt.tight_layout()


plt.show()
