import numpy as np
from scipy.optimize import leastsq, least_squares
from matplotlib import pyplot as plt
import cv2
from module_load_exr import load_exr


def extract_colors_from_cc(img: np.ndarray):
    mat_extracted_cc = np.zeros((4, 6, 3))
    shape_trans = img.shape
    _width_area = np.min(shape_trans[:2]) * 0.35 / 6

    img_copy = img.copy()

    for _i1 in range(4):
        for _i2 in range(6):
            _x = shape_trans[1] * (_i2 + 0.5) / 6
            _y = shape_trans[0] * (_i1 + 0.5) / 4
            _img = img[
                int(_y - _width_area / 2):int(_y - _width_area / 2 + _width_area),
                int(_x - _width_area / 2):int(_x - _width_area / 2 + _width_area), :]
            img_copy[
                int(_y - _width_area / 2):int(_y - _width_area / 2 + _width_area),
                int(_x - _width_area / 2):int(_x - _width_area / 2 + _width_area), :] = 0
            mat_extracted_cc[_i1, _i2, :] = np.mean(_img, axis=(0, 1))

    plt.figure()
    plt.imshow(img_copy)

    return mat_extracted_cc


def transform_pers(img, pts_src, pts_dst, shape_trans):
    mat_pers = cv2.getPerspectiveTransform(pts_src, pts_dst)
    img_trans = cv2.warpPerspective(img, mat_pers, shape_trans)
    return img_trans


# def func_fit(q: np.ndarray, beta, mat_M, w_avg, list_srl, mat_cc):
#     mat_q = q.reshape((3, 3))
#     val: np.ndarray = np.zeros((3, 24))
#     for _i in range(len(list_srl)):
#         _buf = np.dot(mat_q, list_srl[_i])
#         _buf = np.dot(_buf, mat_M)
#         _buf = np.dot(_buf, w_avg[:, None]) / beta
#         val[:, _i] = _buf.ravel() - mat_cc[:, _i]
#     return val.flatten()


def func_fit(q: np.ndarray, list_mat_s, mat_cc, w_avg):
    mat_q = q.reshape((3, 3))
    val: np.ndarray = np.zeros((3, 24))

    for _i in range(len(list_mat_s)):
        # val[:, _i] = (list_mat_s[_i] @ mat_q @ w_avg[:, None] - mat_cc[:, _i:_i + 1]).flatten()
        val[:, _i] = (list_mat_s[_i] @ mat_q @ np.ones((3, 1)) - mat_cc[:, _i:_i + 1]).flatten()

    return val.flatten()


img_cc_w = load_exr(r"C:\Dropbox\TOEI\20240731_OuterCal\CAL_OuterFrustum_v01\CAL_OuterFrustum_v01_fW.exr")
img_cc_r = load_exr(r"C:\Dropbox\TOEI\20240731_OuterCal\CAL_OuterFrustum_v01\CAL_OuterFrustum_v01_fR.exr")
img_cc_g = load_exr(r"C:\Dropbox\TOEI\20240731_OuterCal\CAL_OuterFrustum_v01\CAL_OuterFrustum_v01_fG.exr")
img_cc_b = load_exr(r"C:\Dropbox\TOEI\20240731_OuterCal\CAL_OuterFrustum_v01\CAL_OuterFrustum_v01_fB.exr")
img_cc_refer = load_exr(r"C:\Dropbox\TOEI\20240731_OuterCal\CAL_OuterFrustum_v01\DCI-P3_ColorChecker2014.exr")

shape_trans = int(img_cc_w.shape[1]), int(img_cc_w.shape[0])

pts_src = np.float32(
    [[131, 170], [741, 177], [124, 566], [736, 578]])
pts_dst = np.float32(
    [[0, 0], [shape_trans[0], 0], [0, shape_trans[1]], [shape_trans[0], shape_trans[1]]])
img_trans_w = transform_pers(img_cc_w, pts_src, pts_dst, shape_trans)
img_trans_r = transform_pers(img_cc_r, pts_src, pts_dst, shape_trans)
img_trans_g = transform_pers(img_cc_g, pts_src, pts_dst, shape_trans)
img_trans_b = transform_pers(img_cc_b, pts_src, pts_dst, shape_trans)

mat_extracted_w = extract_colors_from_cc(img_trans_w)
mat_extracted_r = extract_colors_from_cc(img_trans_r)
mat_extracted_g = extract_colors_from_cc(img_trans_g)
mat_extracted_b = extract_colors_from_cc(img_trans_b)
mat_extracted_ref = extract_colors_from_cc(img_cc_refer)

# mat_m = np.array([
#     [1.02806194, -0.01334474, 0.09171636],
#     [-0.03979622, 1.05475851, 0.04785978],
#     [-0.09375769, 0.06229568, 1.14047633]])

# beta = .311
beta = 1.0
mat_cc = mat_extracted_ref.reshape((1, 24, 3)).squeeze().T
vec_rgb_w = mat_extracted_w.reshape((1, 24, 3)).squeeze().T
vec_rgb_r = mat_extracted_r.reshape((1, 24, 3)).squeeze().T
vec_rgb_g = mat_extracted_g.reshape((1, 24, 3)).squeeze().T
vec_rgb_b = mat_extracted_b.reshape((1, 24, 3)).squeeze().T

w_avg = vec_rgb_w[:, 18]

list_mat_s = []
for _i in range(24):
    list_mat_s.append(np.concatenate([
        vec_rgb_r[:, _i],
        vec_rgb_g[:, _i],
        vec_rgb_b[:, _i],
    ], axis=0).reshape((3, 3)))

q_initiate = np.eye(3).reshape(9)

# q_flat, cov = leastsq(func_fit, q_initiate, args=(list_mat_s, mat_cc, w_avg))
param = least_squares(func_fit, q_initiate, bounds=(-2, 2), args=(list_mat_s, mat_cc, w_avg))
q_flat = param.x
mat_q = q_flat.reshape((3, 3))

img_calibrated = func_fit(q_flat, list_mat_s, np.zeros_like(mat_cc), w_avg).reshape((3, 4, 6)).transpose((1, 2, 0))
# img_calibrated = func_fit(np.eye(3), list_mat_s, np.zeros_like(mat_cc), w_avg).reshape((3, 4, 6)).transpose((1, 2, 0))

norm_max = np.max([mat_extracted_ref.max(), mat_extracted_w.max(), img_calibrated.max()])

# plt.figure()
# plt.subplot(2, 3, 1)
# plt.imshow(img_cc_w)
# plt.title("Input Image (White)")
# plt.subplot(2, 3, 2)
# plt.imshow(img_trans_w)
# plt.title("Transformed Image (White)")
# plt.subplot(2, 3, 3)
# plt.imshow(mat_extracted_w)
# plt.title("Extracted Color (White)")
# plt.subplot(2, 3, 4)
# plt.imshow(img_cc_refer)
# plt.title("Extracted Color (White)")
# plt.subplot(2, 3, 6)
# plt.imshow(mat_extracted_ref)
# plt.title("Extracted Color (White)")
# plt.tight_layout()

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img_cc_w)
plt.title("Input Image (White)")
plt.subplot(2, 2, 2)
plt.imshow(mat_extracted_w)
plt.title("Extracted Color (White)")
plt.subplot(2, 2, 3)
plt.imshow(img_cc_refer)
plt.title("ColorChecker2014")
plt.subplot(2, 2, 4)
plt.imshow(mat_extracted_ref)
plt.title("Extracted Color")
plt.suptitle("Color Extraction")
plt.tight_layout()

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(mat_extracted_ref / norm_max)
plt.title("ColorChecker2014")
plt.subplot(2, 2, 3)
plt.imshow(mat_extracted_w / norm_max)
plt.title("From Actual Image")
plt.subplot(2, 2, 4)
plt.imshow(img_calibrated / norm_max)
plt.title("Calibrated")
plt.suptitle("Color Calibration")
plt.tight_layout()

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(mat_extracted_ref)
plt.title("ColorChecker2014")
plt.subplot(2, 2, 3)
plt.imshow(mat_extracted_w)
plt.title("From Actual Image")
plt.subplot(2, 2, 4)
plt.imshow(img_calibrated)
plt.title("Calibrated")
plt.suptitle("Color Calibration")
plt.tight_layout()

print("matrix Q:\n", mat_q)


plt.show()
