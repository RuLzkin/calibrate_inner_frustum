import numpy as np
from matplotlib import pyplot as plt
from modules.module_load_exr import load_exr
from modules.module_inner_calibration import calculate_matrix_m, comparison

img_input = load_exr(r"C:\Dropbox\TOEI\20240718_check_cms\New_CMStestpattern_source.exr")
img_output = load_exr(r"C:\Dropbox\TOEI\20240924_inner_correction\ShootAltCam_DCI-P3_CMS_forInnerCalibration.exr")

print(img_input.shape)
print(img_output.shape)

mat_cms_input = np.flipud(img_input)
mat_cms_output = np.flipud(img_output)

vec_cms_input = mat_cms_input.reshape((15 * 15, 3))[:6 * 6 * 6, :]
vec_cms_output = mat_cms_output.reshape((15 * 15, 3))[:6 * 6 * 6, :]

print(vec_cms_input.shape)

matrix_m = calculate_matrix_m(vec_cms_input.T, vec_cms_output.T)

np.set_printoptions(precision=8, suppress=True)
print(matrix_m)

vec_calib = np.dot(matrix_m, vec_cms_output.T).T

comparison(vec_cms_input, vec_cms_output, "not calibrated")
comparison(vec_cms_input, vec_calib, "calibrated (simu.)")

plt.show()
