import numpy as np
from matplotlib import pyplot as plt
from module_load_exr import load_exr, method_okawa


def comparison(vec_cms_input, vec_cms_output, suptitle):

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


img_input = load_exr("C:/Dropbox/TOEI/20231026_test_pattern/CMS_TestPattern.exr")
img_ovpc = load_exr(r"C:\Dropbox\TOEI\20240718_check_cms\Shoot_CalibratedCMS_OpenVPCal.exr")
img_okawa = load_exr(r"C:\Dropbox\TOEI\20240718_check_cms\Shoot_CalibratedCMS_Okawa.exr")
# img_output = load_exr(r"C:\Dropbox\TOEI\20240708_OpenVPCal\Shoot_Check_OpenVPCal_AP0.exr")


img_input = np.flipud(img_input)

mat_cms_input = np.zeros((15, 15, 3))
mat_cms_ovpc = np.flipud(img_ovpc)
mat_cms_okaw = np.flipud(img_okawa)

for _i in range(15):
    for _j in range(15):
        _ii, _jj = 7 * _i, 7 * _j
        mat_cms_input[_i, _j, :] = img_input[_ii:_ii + 7, _jj:_jj + 7, :].mean(axis=(0, 1))

vec_cms_input = mat_cms_input.reshape((15 * 15, 3))[:6 * 6 * 6, :]
vec_cms_ovpc = mat_cms_ovpc.reshape((15 * 15, 3))[:6 * 6 * 6, :]
vec_cms_okaw = mat_cms_okaw.reshape((15 * 15, 3))[:6 * 6 * 6, :]

matrix_ovpc = method_okawa(vec_cms_input.T, vec_cms_ovpc.T)
matrix_okaw = method_okawa(vec_cms_input.T, vec_cms_okaw.T)

print(matrix_ovpc)
print(matrix_okaw)


comparison(vec_cms_input, vec_cms_ovpc, "OpenVPCal")
comparison(vec_cms_input, vec_cms_okaw, "Our")

plt.show()
