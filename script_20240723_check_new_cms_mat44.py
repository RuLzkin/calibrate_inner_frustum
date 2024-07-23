import numpy as np
from matplotlib import pyplot as plt
from module_load_exr import load_exr, method_okawa_newcms


def comparison(vec_cms_input, vec_cms_output, suptitle):

    plt.figure(figsize=(9, 6))
    plt.subplot(3, 2, 1)
    plt.plot(vec_cms_input[:, 0], "r", label="Red")
    plt.plot(vec_cms_input[:, 1], "g", label="Green")
    plt.plot(vec_cms_input[:, 2], "b", label="Blue")
    plt.ylim(-2, 12)
    plt.grid()
    plt.legend()
    plt.xlabel("CMS Patch")
    plt.ylabel("Intensity")
    plt.title("Input")
    plt.subplot(3, 2, 3)
    plt.plot(vec_cms_output[:, 0], "r", label="Red")
    plt.plot(vec_cms_output[:, 1], "g", label="Green")
    plt.plot(vec_cms_output[:, 2], "b", label="Blue")
    plt.ylim(-2, 12)
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
        plt.ylim(-2, 12)
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
                plt.plot([-2, 12], [-2, 12], "--")
            plt.xlim(-2, 12)
            plt.ylim(-2, 12)
            plt.grid()
            plt.xlabel("Input: " + _color_i)
            plt.ylabel("Output: " + _color_j)
            # plt.legend()
    plt.suptitle(suptitle)
    plt.tight_layout()


img_input = load_exr(r"C:\Dropbox\TOEI\20240718_check_cms\New_CMStestpattern_source.exr")
img_output = load_exr(r"C:\Dropbox\TOEI\20240723_check_cms\Shoot_DCI-P3_newCMS_forInnerCalibration.exr")
# img_output = load_exr(r"C:\Dropbox\TOEI\20240718_check_cms\New_CMStestpattern_source.exr")

print(img_input.shape)
print(img_output.shape)

mat_cms_input = np.flipud(img_input)
mat_cms_output = np.flipud(img_output)

vec_cms_input = mat_cms_input.reshape((15 * 15, 3))[:6 * 6 * 6, :]
vec_cms_output = mat_cms_output.reshape((15 * 15, 3))[:6 * 6 * 6, :]

print(vec_cms_input.shape)

vec_cms_input = np.concatenate([vec_cms_input, np.ones((6 * 6 * 6, 1))], axis=1)
vec_cms_output = np.concatenate([vec_cms_output, np.ones((6 * 6 * 6, 1))], axis=1)

matrix_okawa = method_okawa_newcms(vec_cms_input.T, vec_cms_output.T)

np.set_printoptions(precision=8, suppress=True)
print(matrix_okawa)

# vec_calib = np.maximum(0, np.dot(matrix_okawa, vec_cms_output.T).T)
vec_calib = np.dot(matrix_okawa, vec_cms_output.T).T

comparison(vec_cms_input[:, :-1], vec_cms_output[:, :-1], "not calibrated")
comparison(vec_cms_input[:, :-1], vec_calib[:, :-1], "calibrated (simu.)")

plt.show()
