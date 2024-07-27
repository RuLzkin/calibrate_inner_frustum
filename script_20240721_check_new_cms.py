import numpy as np
from matplotlib import pyplot as plt
from module_load_exr import load_exr, method_okawa


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
# img_output = load_exr(r"C:\Dropbox\TOEI\20240708_OpenVPCal\Shoot_Check_OpenVPCal_AP0.exr")

print(img_input.shape)

img_input = np.flipud(img_input)

vec_cms_input = img_input.reshape((15 * 15, 3))[:6 * 6 * 6, :]

matrix_okawa = method_okawa(vec_cms_input.T, vec_cms_input.T)

print(matrix_okawa)
print(np.max(img_input))
print(np.max(vec_cms_input))

for _i in range(3):
    plt.figure()
    plt.imshow(img_input[..., _i])

comparison(vec_cms_input, vec_cms_input, "New CMS")

plt.show()
