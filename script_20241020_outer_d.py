import numpy as np
from matplotlib import pyplot as plt
from modules.module_load_exr import load_exr
from modules.module_extract_cc_gui import ColorcheckerExtractor
from modules.module_outer_calibration import (
    calculate_matrix_q,
    comparison,
    func_fit,
    colorchecker2colors)

img_cc_w = load_exr(r"C:\Dropbox\TOEI\20240731_OuterCal\CAL_OuterFrustum_v01\CAL_OuterFrustum_v01_dW.exr")
img_cc_r = load_exr(r"C:\Dropbox\TOEI\20240731_OuterCal\CAL_OuterFrustum_v01\CAL_OuterFrustum_v01_dR.exr")
img_cc_g = load_exr(r"C:\Dropbox\TOEI\20240731_OuterCal\CAL_OuterFrustum_v01\CAL_OuterFrustum_v01_dG.exr")
img_cc_b = load_exr(r"C:\Dropbox\TOEI\20240731_OuterCal\CAL_OuterFrustum_v01\CAL_OuterFrustum_v01_dB.exr")
img_cc_t = load_exr(r"C:\Dropbox\TOEI\20240806_OUterCal_check\OuterCheck_Ref.exr")

processor = ColorcheckerExtractor(img_cc_w)
pts_src = processor.get_four_points()
processor = ColorcheckerExtractor(img_cc_t)
pts_src_t = processor.get_four_points()

shape_trans = processor.shape_transform()
pts_dst = processor.coords_dst()

mat_extracted_w = colorchecker2colors(img_cc_w, pts_src, pts_dst, shape_trans, "CAL_OuterFrustum_v01_dW.exr")
mat_extracted_r = colorchecker2colors(img_cc_r, pts_src, pts_dst, shape_trans, "CAL_OuterFrustum_v01_dR.exr")
mat_extracted_g = colorchecker2colors(img_cc_g, pts_src, pts_dst, shape_trans, "CAL_OuterFrustum_v01_dG.exr")
mat_extracted_b = colorchecker2colors(img_cc_b, pts_src, pts_dst, shape_trans, "CAL_OuterFrustum_v01_dB.exr")
mat_extracted_t = colorchecker2colors(img_cc_t, pts_src_t, pts_dst, shape_trans, "OuterCheck_Ref.exr")

mat_extracted_t = mat_extracted_t / np.max(mat_extracted_t)

mat_cc = mat_extracted_t.reshape((1, 24, 3)).squeeze().T
vec_rgb_w = mat_extracted_w.reshape((1, 24, 3)).squeeze().T
vec_rgb_r = mat_extracted_r.reshape((1, 24, 3)).squeeze().T
vec_rgb_g = mat_extracted_g.reshape((1, 24, 3)).squeeze().T
vec_rgb_b = mat_extracted_b.reshape((1, 24, 3)).squeeze().T

w_avg = mat_cc[:, 18]

list_mat_s = []
for _i in range(24):
    list_mat_s.append(np.concatenate([
        vec_rgb_r[:, _i],
        vec_rgb_g[:, _i],
        vec_rgb_b[:, _i],
    ], axis=0).reshape((3, 3)))

mat_q = calculate_matrix_q(list_mat_s, mat_cc, w_avg)

img_calibrated = func_fit(mat_q.ravel(), list_mat_s, np.zeros_like(mat_cc), w_avg).reshape((3, 4, 6)).transpose((1, 2, 0))

comparison(mat_extracted_t, mat_extracted_w, img_cc_w, img_cc_t, img_calibrated)

np.set_printoptions(precision=3, suppress=True)
print("norm(before)\n", np.linalg.norm(mat_extracted_w - mat_extracted_t, axis=2))
print("norm(after)\n", np.linalg.norm(img_calibrated - mat_extracted_t, axis=2))

print("diff norm\n", np.linalg.norm(img_calibrated - mat_extracted_t, axis=2) - np.linalg.norm(mat_extracted_w - mat_extracted_t, axis=2))

print("norm(before)", np.linalg.norm(mat_extracted_w - mat_extracted_t))
print("norm(after) ", np.linalg.norm(img_calibrated - mat_extracted_t))

print("matrix Q:\n", mat_q)

plt.show()
