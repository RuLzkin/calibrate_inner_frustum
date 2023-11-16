import numpy as np
from matplotlib import pyplot as plt
from module_load_exr import load_exr, method_okawa, method_ue


def comparison(vec_cms_input, vec_cms_output, suptitle):
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.imshow(img_input)
    # plt.subplot(2, 2, 2)
    # plt.imshow(img_output)
    # plt.subplot(2, 2, 3)
    # plt.imshow(mat_cms_input)
    # plt.subplot(2, 2, 4)
    # plt.imshow(mat_cms_output)

    plt.figure(figsize=(9, 6))
    plt.subplot(3, 2, 1)
    plt.plot(vec_cms_input[:, 0], "r", label="Red")
    plt.plot(vec_cms_input[:, 1], "g", label="Green")
    plt.plot(vec_cms_input[:, 2], "b", label="Blue")
    plt.ylim(-0.2, 1.2)
    plt.grid()
    plt.legend()
    plt.xlabel("CMS Patch")
    plt.ylabel("Intensity")
    plt.title("Input")
    plt.subplot(3, 2, 3)
    plt.plot(vec_cms_output[:, 0], "r", label="Red")
    plt.plot(vec_cms_output[:, 1], "g", label="Green")
    plt.plot(vec_cms_output[:, 2], "b", label="Blue")
    plt.ylim(-0.2, 1.2)
    plt.grid()
    plt.legend()
    plt.xlabel("CMS Patch")
    plt.ylabel("Intensity")
    plt.title("Output")
    plt.suptitle(suptitle)
    plt.tight_layout()

    # plt.figure(figsize=(5, 6))
    for _i, _color_i in enumerate(["red", "green", "blue"]):
        plt.subplot(3, 2, 2 * (_i + 1))
        plt.plot(vec_cms_input[:, _i], label="input")
        plt.plot(vec_cms_output[:, _i], "--", label="output")
        plt.title(_color_i)
        plt.ylim(-0.2, 1.2)
        plt.grid()
        plt.legend()
        plt.xlabel("CMS Patch")
        plt.ylabel("Output")
    plt.suptitle(suptitle)
    plt.tight_layout()

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(ten_cms_input[:, :, 0, :].squeeze())
    # plt.subplot(1, 2, 2)
    # plt.imshow(ten_cms_output[:, :, 0, :].squeeze())

    # plt.figure(figsize=(3, 8))
    # for _i, _color_i in enumerate(["red", "green", "blue"]):
    #     plt.subplot(3, 1, _i + 1)
    #     plt.scatter(vec_cms_input[:, _i], vec_cms_output[:, _i])
    #     plt.plot([-0.2, 1.2], [-0.2, 1.2], "--")
    #     plt.title(_color_i)
    #     plt.xlim(-0.2, 1.2)
    #     plt.ylim(-0.2, 1.2)
    #     plt.grid()
    #     plt.xlabel("Input")
    #     plt.ylabel("Output")
    #     # plt.legend()
    # plt.tight_layout()

    plt.figure(figsize=(8, 8))
    for _i, _color_i in enumerate(["red", "green", "blue"]):
        for _j, _color_j in enumerate(["red", "green", "blue"]):
            plt.subplot(3, 3, _i + 3 * _j + 1)
            plt.scatter(vec_cms_input[:, _i], vec_cms_output[:, _j])
            if _i == _j:
                plt.plot([-0.2, 1.2], [-0.2, 1.2], "--")
            plt.xlim(-0.2, 1.2)
            plt.ylim(-0.2, 1.2)
            plt.grid()
            plt.xlabel("Input: " + _color_i)
            plt.ylabel("Output: " + _color_j)
            # plt.legend()
    plt.suptitle(suptitle)
    plt.tight_layout()


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


matrix_okawa = method_okawa(vec_cms_input.T, vec_cms_output.T)

print(matrix_okawa)

# print(np.sum(np.abs(vec_cms_input.T - np.dot(matrix_okawa, vec_cms_output.T))))
# print(np.sum(np.abs(vec_cms_input.T - vec_cms_output.T)))
# print(np.sum(np.abs(vec_cms_input.T)))
# print(np.sum(np.abs(vec_cms_output.T)))

vec_calib = np.maximum(0, np.dot(matrix_okawa, vec_cms_output.T).T)

comparison(vec_cms_input, vec_cms_output, "not calibrated")
comparison(vec_cms_input, vec_calib, "calibrated")

plt.show()
