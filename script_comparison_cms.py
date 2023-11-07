import numpy as np
from matplotlib import pyplot as plt
from module_load_exr import load_exr

img_input = load_exr("./testfiles/CMS_TestPattern.exr")
img_output = load_exr("./testfiles/CMS_Shoot_EOTF_Corrected.exr")

print(img_input.shape)
print(img_output.shape)

img_input = np.flipud(img_input)
img_output = np.flipud(img_output)

mat_cms_input = np.zeros((15, 15, 3))
mat_cms_output = np.zeros((15, 15, 3))

for _i in range(15):
    for _j in range(15):
        _ii, _jj = 7 * _i, 7 * _j
        mat_cms_input[_i, _j, :] = img_input[_ii:_ii + 7, _jj:_jj + 7, :].mean(axis=(0, 1))

        _i_stt = int(512 * (_i + 0.25) / 15)
        _i_end = int(512 * (_i + 0.75) / 15)
        _j_stt = int(512 * (_j + 0.25) / 15)
        _j_end = int(512 * (_j + 0.75) / 15)
        mat_cms_output[_i, _j, :] = img_output[_i_stt:_i_end, _j_stt:_j_end, :].mean(axis=(0, 1))


vec_cms_input = mat_cms_input.reshape((15 * 15, 3))[:6 * 6 * 6, :]
vec_cms_output = mat_cms_output.reshape((15 * 15, 3))[:6 * 6 * 6, :]

ten_cms_input = np.zeros((6, 6, 6, 3))
ten_cms_output = np.zeros((6, 6, 6, 3))
for _count in range(6 * 6 * 6):
    _i = _count % 6
    _j = np.floor(_count / 6).astype(int) % 6
    _k = np.floor(_count / 36).astype(int)
    ten_cms_input[_i, _j, _k, :] = vec_cms_input[_count, :]
    ten_cms_output[_i, _j, _k, :] = vec_cms_output[_count, :]


plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img_input)
plt.subplot(2, 2, 2)
plt.imshow(img_output)
plt.subplot(2, 2, 3)
plt.imshow(mat_cms_input)
plt.subplot(2, 2, 4)
plt.imshow(mat_cms_output)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(vec_cms_input)
plt.ylim(-0.2, 1.2)
plt.subplot(1, 2, 2)
plt.plot(vec_cms_output)
plt.ylim(-0.2, 1.2)

plt.figure()
for _i, _color in enumerate(["red", "green", "blue"]):
    plt.subplot(2, 2, _i + 1)
    plt.plot(vec_cms_input[:, _i], label="input")
    plt.plot(vec_cms_output[:, _i], "--", label="output")
    plt.title(_color)
    plt.ylim(-0.2, 1.2)
    plt.grid()
    plt.legend()
plt.tight_layout()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(ten_cms_input[:, :, 0, :].squeeze())
plt.subplot(1, 2, 2)
plt.imshow(ten_cms_output[:, :, 0, :].squeeze())

plt.figure()
for _i, _color in enumerate(["red", "green", "blue"]):
    plt.subplot(2, 2, _i + 1)
    plt.scatter(vec_cms_input[:, _i], vec_cms_output[:, _i])
    plt.title(_color)
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.grid()
    # plt.legend()

B = np.dot(vec_cms_input.T, np.linalg.pinv(vec_cms_output.T))

print(np.sum(np.abs(vec_cms_input.T - np.dot(B, vec_cms_output.T))) / np.sum(np.abs(vec_cms_input)))

plt.show()
