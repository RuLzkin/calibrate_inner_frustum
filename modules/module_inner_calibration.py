import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from module_load_exr import load_exr


def calculate_matrix_m(input: ndarray, output: ndarray) -> ndarray:
    """Inner calibration

        Args:
            input: np.ndarray, A 15 * 15 * 3 array input to the LED.
            output: np.ndarray, A 15 * 15 * 3 array output from the camera.

        Returns:
            matrix M: ndarray, A calculated 3x3 matrix M for pre-calibration.

        Examples:
            >>> matrix_m = calculate_matrix_m(input, output)
        """
    buf_output: ndarray = output.copy()
    buf_return: ndarray = np.linalg.inv(np.dot(buf_output, np.linalg.pinv(input)))

    # 4 * 4 matrix
    if buf_return.shape[0] > 3:
        buf_return[3, :] = [0, 0, 0, 1]

    return buf_return


def path_to_matrix_m(path_input: str, path_output: str, show_comparison: bool = False) -> ndarray:
    """Function that encapsulates all steps of the Inner Calibration process.

        Args:
            path_input: np.ndarray, filepath input to the LED
            path_output: np.ndarray, filepath output from the camera.
            show_comparison: bool, flag to show comparison

        Returns:
            matrix M: ndarray, A calculated 3x3 matrix M for pre-calibration.

        Examples:
            >>> matrix_m = path_to_matrix_m(
                    "New_CMStestpattern_source.exr",
                    "ShootAltCam_DCI-P3_CMS_forInnerCalibration.exr")
    """
    img_input = load_exr(path_input)
    img_output = load_exr(path_output)

    mat_cms_input = np.flipud(img_input)
    mat_cms_output = np.flipud(img_output)

    vec_cms_input = mat_cms_input.reshape((15 * 15, 3))[:6 * 6 * 6, :]
    vec_cms_output = mat_cms_output.reshape((15 * 15, 3))[:6 * 6 * 6, :]

    matrix_m: ndarray = calculate_matrix_m(vec_cms_input.T, vec_cms_output.T)

    if show_comparison:
        vec_calib = np.dot(matrix_m, vec_cms_output.T).T
        comparison(vec_cms_input, vec_cms_output, "not calibrated")
        comparison(vec_cms_input, vec_calib, "calibrated (simu.)")

    return matrix_m


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


if __name__ == "__main__":
    """ usage 1 """
    matrix_m = path_to_matrix_m(
        "New_CMStestpattern_source.exr",
        "ShootAltCam_DCI-P3_CMS_forInnerCalibration.exr")

    """ usage 2 """
    img_input = load_exr("New_CMStestpattern_source.exr")
    img_output = load_exr("ShootAltCam_DCI-P3_CMS_forInnerCalibration.exr")

    mat_cms_input = np.flipud(img_input)
    mat_cms_output = np.flipud(img_output)

    vec_cms_input = mat_cms_input.reshape((15 * 15, 3))[:6 * 6 * 6, :]
    vec_cms_output = mat_cms_output.reshape((15 * 15, 3))[:6 * 6 * 6, :]

    matrix_m = calculate_matrix_m(vec_cms_input.T, vec_cms_output.T)

    np.set_printoptions(precision=8, suppress=True)  # for print function

    vec_calib = np.dot(matrix_m, vec_cms_output.T).T

    comparison(vec_cms_input, vec_cms_output, "not calibrated")
    comparison(vec_cms_input, vec_calib, "calibrated (simu.)")

    print("matrix M:", matrix_m)
    print("norm(before)", np.linalg.norm(vec_cms_input - vec_cms_output))
    print("norm(after) ", np.linalg.norm(vec_cms_input - vec_calib))

    plt.show()
