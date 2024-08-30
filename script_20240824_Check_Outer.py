import numpy as np
import numpy.linalg as la
from scipy.optimize import leastsq, least_squares
from matplotlib import pyplot as plt
import cv2
from module_load_exr import load_exr, preview_exr


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


# preview_exr(r"C:\Dropbox\TOEI\20240820_OuterCheck\OuterCheck_M.exr")
img_cc_m = load_exr(r"C:\Dropbox\TOEI\20240820_OuterCheck\OuterCheck_M.exr")
img_cc_d = load_exr(r"C:\Dropbox\TOEI\20240820_OuterCheck\OuterCheck_Dre.exr")
img_cc_f = load_exr(r"C:\Dropbox\TOEI\20240820_OuterCheck\OuterCheck_Fre.exr")
img_cc_refer = load_exr(r"C:\Dropbox\TOEI\20240731_OuterCal\CAL_OuterFrustum_v01\DCI-P3_ColorChecker2014.exr")

shape_trans = int(img_cc_m.shape[1]), int(img_cc_m.shape[0])

pts_src = np.float32(
    [[2153, 1821], [2575, 1823], [2155, 2100], [2578, 2105]])
pts_dst = np.float32(
    [[0, 0], [shape_trans[0], 0], [0, shape_trans[1]], [shape_trans[0], shape_trans[1]]])

img_trans_m = transform_pers(img_cc_m, pts_src, pts_dst, shape_trans)
img_trans_d = transform_pers(img_cc_d, pts_src, pts_dst, shape_trans)
img_trans_f = transform_pers(img_cc_f, pts_src, pts_dst, shape_trans)

mat_extracted_ref = extract_colors_from_cc(img_cc_refer)
mat_extracted_m = extract_colors_from_cc(img_trans_m)
mat_extracted_d = extract_colors_from_cc(img_trans_d)
mat_extracted_f = extract_colors_from_cc(img_trans_f)

white_ref = la.norm(mat_extracted_ref[3, 0, :])
white_m = la.norm(mat_extracted_m[3, 0, :])
white_d = la.norm(mat_extracted_d[3, 0, :])
white_f = la.norm(mat_extracted_f[3, 0, :])

print(np.array2string(la.norm(mat_extracted_m / white_m - mat_extracted_ref / white_ref, axis=2), precision=3, suppress_small=True))
print(np.array2string(la.norm(mat_extracted_d / white_d - mat_extracted_ref / white_ref, axis=2), precision=3, suppress_small=True))
print(np.array2string(la.norm(mat_extracted_f / white_f - mat_extracted_ref / white_ref, axis=2), precision=3, suppress_small=True))

# plt.show()
