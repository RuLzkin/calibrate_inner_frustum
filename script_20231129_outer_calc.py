import numpy as np
from scipy.optimize import leastsq
from module_load_exr import preview_exr, load_exr
from matplotlib import pyplot as plt

# https://technorgb.blogspot.com/2017/10/blog-post.html

"""WIP"""

img_cc_w = load_exr("C:/Dropbox/TOEI/20231122_Outer_patch/For_OuterCalib_W.258.exr")
img_cc_r = load_exr("C:/Dropbox/TOEI/20231122_Outer_patch/For_OuterCalib_R.31.exr")
img_cc_g = load_exr("C:/Dropbox/TOEI/20231122_Outer_patch/For_OuterCalib_G.76.exr")
img_cc_b = load_exr("C:/Dropbox/TOEI/20231122_Outer_patch/For_OuterCalib_B.184.exr")

iijj_babel = [
    [700, 900, 1575, 1750],  # 1
    [650, 900, 1800, 2000],
    [600, 850, 2050, 2250],
    [550, 800, 2400, 2600],
    [500, 750, 2700, 3000],
    [400, 700, 3100, 3400],  # 6
    [1050, 1250, 1580, 1740],  # 7
    [1000, 1250, 1825, 2000],
    [1000, 1250, 2100, 2300],
    [950, 1250, 2400, 2600],
    [950, 1200, 2700, 3000],
    [900, 1200, 3100, 3400],  # 12
    [1400, 1600, 1560, 1730],  # 13
    [1400, 1600, 1800, 2000],
    [1400, 1600, 2054, 2295],
    [1400, 1600, 2350, 2631],
    [1400, 1600, 2695, 3013],
    [1400, 1600, 3087, 3458],  # 18
    [1850, 1950, 1536, 1727],  # 19
    [1850, 1950, 1781, 1992],
    [1850, 1950, 2045, 2294],
    [1850, 1950, 2348, 2627],
    [1850, 1950, 2687, 3000],
    [1850, 1950, 3080, 3450],  # 24
]

list_rgb_w = []
list_rgb_r = []
list_rgb_g = []
list_rgb_b = []
for _i in range(24):
    _iijj = iijj_babel[_i]
    list_rgb_w.append(img_cc_w[_iijj[0]:_iijj[1], _iijj[2]:_iijj[3], :].mean(axis=(0, 1)))
    list_rgb_r.append(img_cc_r[_iijj[0]:_iijj[1], _iijj[2]:_iijj[3], :].mean(axis=(0, 1)))
    list_rgb_g.append(img_cc_g[_iijj[0]:_iijj[1], _iijj[2]:_iijj[3], :].mean(axis=(0, 1)))
    list_rgb_b.append(img_cc_b[_iijj[0]:_iijj[1], _iijj[2]:_iijj[3], :].mean(axis=(0, 1)))

mat_cc_xyz = np.array([
    [[.1093, .0971, .063], [.3681, .3327, .2443], [.1699, .181, .3371], [.1067, .134, .0683], [.2467, .231, .4349], [.3032, .4171, .4534]],
    [[.3853, .3053, .0669], [.1304, .1128, .3817], [.2804, .1847, .1292], [.085, .0632, .1384], [.3228, .4295, .1062], [.4497, .4214, .077]],
    [[.0769, .059, .2761], [.1345, .2231, .0941], [.2044, .1238, .0515], [.5534, .5901, .0891], [.2955, .1876, .2943], [.1335, .185, .3854]],
    [[.8298, .8802, .9152], [.5583, .59, .6373], [.3447, .3649, .3978], [.1809, .1906, .2068], [.0834, .0882, .0976], [.03, .0315, .0351]],
])

mat_rgb2xyz = np.array([
    [0.490, 0.310, 0.200],
    [0.17697, 0.81240, 0.010630],
    [0.000, 0.010, 0.990]
]) / 0.17697

mat_xyz2rgb = np.linalg.pinv(mat_rgb2xyz)

img_cc_rgb = np.zeros_like(mat_cc_xyz)
for _i in range(4):
    for _j in range(6):
        img_cc_rgb[_i, _j, :] = np.dot(mat_xyz2rgb, np.squeeze(mat_cc_xyz[_i, _j, :])[:, None]).squeeze()

img_captured_w = np.zeros_like(mat_cc_xyz)
_n = 0
for _i in range(4):
    for _j in range(6):
        img_captured_w[_i, _j, :] = list_rgb_w[_n]
        _n += 1

print(mat_cc_xyz.shape)

# preview_exr("C:/Dropbox/TOEI/20231122_Outer_patch/For_OuterCalib_B.184.exr")
# preview_exr("C:/Dropbox/TOEI/20231122_Outer_patch/For_OuterCalib_G.76.exr")
# preview_exr("C:/Dropbox/TOEI/20231122_Outer_patch/For_OuterCalib_R.31.exr")
preview_exr("C:/Dropbox/TOEI/20231122_Outer_patch/For_OuterCalib_W.258.exr", show=False, amp=5)


w_avg = list_rgb_w[18]
mat_M = np.eye(3)
beta = .0311
mat_cc = img_cc_rgb.reshape((1, 24, 3)).squeeze().T

list_srl = []
for _i in range(24):
    list_srl.append(np.concatenate([
        list_rgb_r[_i][None, :],
        list_rgb_g[_i][None, :],
        list_rgb_b[_i][None, :],
    ]))


def func_fit(q: np.ndarray, beta, mat_M, w_avg, list_srl, mat_cc):
    mat_q = q.reshape((3, 3))
    val: np.ndarray = np.zeros((3, 24))
    for _i in range(len(list_srl)):
        _buf = np.dot(mat_q, list_srl[_i])
        _buf = np.dot(_buf, mat_M)
        _buf = np.dot(_buf, w_avg[:, None]) / beta
        val[:, _i] = _buf.ravel() - mat_cc[:, _i]
    return val.flatten()


q_initiate = np.eye(3).reshape(9)

q_flat, cov = leastsq(func_fit, q_initiate, args=(beta, mat_M, w_avg, list_srl, mat_cc))
mat_q = q_flat.reshape((3, 3))

img_calibrated = np.zeros_like(img_cc_rgb)
_n = 0
for _i in range(4):
    for _j in range(6):
        _buf = np.dot(mat_q, list_srl[_n])
        _buf = np.dot(_buf, mat_M)
        _buf = np.dot(_buf, w_avg[:, None]) / beta
        img_calibrated[_i, _j, :] = _buf.ravel()
        _n += 1

norm_max = np.max([img_cc_rgb.max(), img_captured_w.max(), img_calibrated.max()])

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img_cc_rgb / norm_max)
plt.title("From X-Rite")
plt.subplot(2, 2, 3)
plt.imshow(img_captured_w / norm_max)
plt.title("From Actual Image")
plt.subplot(2, 2, 4)
plt.imshow(img_calibrated / norm_max)
plt.title("Calibrated")
plt.tight_layout()

print(mat_q)

plt.show()
