import numpy as np
from numpy import ndarray
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
import cv2
from .module_load_exr import load_exr
from .module_extract_cc_gui import ColorcheckerExtractor


def calculate_matrix_q(list_mat_s: list, mat_cc: ndarray, w_avg: ndarray) -> ndarray:
    """Calculate Matrix Q

        Args:
            list_mat_s: list[ndarray],
                List of matrices, each representing a color patch illuminated by R, G, B light sources. Length equals the number of color checker patches. Each matrix column is a color vector.
            mat_cc: ndarray, RGB values of color checker patches in the target image. 3 rows (R, G, B) and columns equal to the number of patches.
            w_avg: ndarray, RGB values of the whitest patch in the color checker.

        Returns:
            ndarray, 3 * 3 matrix, Q matrix for Outer Frustum

        Examples:
            >>> mat_q = calculate_matrix_q(list_mat_s, mat_cc, w_avg)
            [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
    """
    q_initiate: ndarray = np.eye(3).reshape(9)
    param = least_squares(func_fit, q_initiate, bounds=(-2, 2), args=(list_mat_s, mat_cc, w_avg))
    q_flat: ndarray = param.x
    mat_q: ndarray = q_flat.reshape((3, 3))
    return mat_q


def func_fit(q: np.ndarray, list_mat_s, mat_cc, w_avg) -> ndarray:
    mat_q: ndarray = q.reshape((3, 3))
    val: np.ndarray = np.zeros((3, 24))

    for _i in range(len(list_mat_s)):
        val[:, _i] = (list_mat_s[_i] @ mat_q @ w_avg[:, None] - mat_cc[:, _i:_i + 1]).flatten()

    return val.flatten()


def colorchecker2colors(img_cc, pts_src, pts_dst, shape_trans, str_title="test", show_preview=True) -> ndarray:
    """(wrapper) Extract Colors from Color Checker

    Args:
        img_cc: ndarray, input image
        pts_src, pts_dst: cv2.getPerspectiveTransform(pts_src, pts_dst)
        shape_trans: dsize args for cv2.warpPerspective()
        str_title: str, for preview
        show_preview: bool, show preview or not

    Returns:
        ndarray, shape(row_cc, column_cc, [R, G, B])
    """
    img_trans = transform_pers(img_cc, pts_src, pts_dst, shape_trans)
    mat_extracted = extract_colors_from_cc(img_trans, str_title, show_preview=show_preview)
    return mat_extracted


def extract_colors_from_cc(img: ndarray, str_title: str = "", show_preview: bool = False) -> ndarray:
    mat_extracted_cc: ndarray = np.zeros((4, 6, 3))
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

    if show_preview:
        plt.figure()
        plt.imshow(img_copy)
        plt.title(str_title)

    return mat_extracted_cc


def transform_pers(img, pts_src, pts_dst, shape_trans) -> ndarray:
    mat_pers: ndarray = cv2.getPerspectiveTransform(pts_src, pts_dst)
    img_trans: ndarray = cv2.warpPerspective(img, mat_pers, shape_trans)
    return img_trans


def comparison(mat_extracted_ref, mat_extracted_w, img_cc_w, img_cc_refer, img_calibrated):

    norm_max = np.max([mat_extracted_ref.max(), mat_extracted_w.max(), img_calibrated.max()])

    key_kron = np.zeros((2, 2, 1))
    key_kron[:, 0, :] = 1

    img_comp = np.kron(mat_extracted_ref, key_kron) + np.kron(img_calibrated, 1 - key_kron)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img_cc_w)
    plt.title("Input Image (White)")
    plt.subplot(2, 2, 2)
    plt.imshow(mat_extracted_w)
    plt.title("Extracted Color (White)")
    plt.subplot(2, 2, 3)
    plt.imshow(img_cc_refer)
    plt.title("ColorChecker (Ref.)")
    plt.subplot(2, 2, 4)
    plt.imshow(mat_extracted_ref)
    plt.title("Extracted Color")
    plt.suptitle("Color Extraction")
    plt.tight_layout()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(mat_extracted_ref / norm_max)
    plt.title("ColorChecker (Ref.)")
    plt.subplot(2, 2, 2)
    plt.imshow(img_comp / norm_max)
    plt.title("Comparison (Ref. / Calib.)")
    plt.subplot(2, 2, 3)
    plt.imshow(mat_extracted_w / norm_max)
    plt.title("From Actual Image")
    plt.subplot(2, 2, 4)
    plt.imshow(img_calibrated / norm_max)
    plt.title("Calibrated")
    plt.suptitle("Color Calibration (Norm.)")
    plt.tight_layout()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(mat_extracted_ref)
    plt.title("ColorChecker (Ref.)")
    plt.subplot(2, 2, 2)
    plt.imshow(img_comp)
    plt.title("Comparison (Ref. / Calib.)")
    plt.subplot(2, 2, 3)
    plt.imshow(mat_extracted_w)
    plt.title("From Actual Image")
    plt.subplot(2, 2, 4)
    plt.imshow(img_calibrated)
    plt.title("Calibrated")
    plt.suptitle("Color Calibration")
    plt.tight_layout()


if __name__ == "__main__":

    img_cc_w = load_exr("CAL_OuterFrustum_v01_dW.exr")
    img_cc_r = load_exr("CAL_OuterFrustum_v01_dR.exr")
    img_cc_g = load_exr("CAL_OuterFrustum_v01_dG.exr")
    img_cc_b = load_exr("CAL_OuterFrustum_v01_dB.exr")
    img_cc_t = load_exr("OuterCheck_Ref.exr")

    processor = ColorcheckerExtractor(img_cc_w)
    pts_src = processor.get_four_points()
    processor = ColorcheckerExtractor(img_cc_t)
    pts_src_ref = processor.get_four_points()

    shape_trans = processor.shape_transform()
    pts_dst = processor.coords_dst()

    mat_extracted_w = colorchecker2colors(img_cc_w, pts_src, pts_dst, shape_trans, "CAL_OuterFrustum_v01_dW.exr", show_preview=True)
    mat_extracted_r = colorchecker2colors(img_cc_r, pts_src, pts_dst, shape_trans, "CAL_OuterFrustum_v01_dR.exr", show_preview=True)
    mat_extracted_g = colorchecker2colors(img_cc_g, pts_src, pts_dst, shape_trans, "CAL_OuterFrustum_v01_dG.exr", show_preview=True)
    mat_extracted_b = colorchecker2colors(img_cc_b, pts_src, pts_dst, shape_trans, "CAL_OuterFrustum_v01_dB.exr", show_preview=True)
    mat_extracted_t = colorchecker2colors(img_cc_t, pts_src_ref, pts_dst, shape_trans, "OuterCheck_Ref.exr", show_preview=True)

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

    print("matrix Q:\n", mat_q)
    print("norm(before)", np.linalg.norm(mat_extracted_w - mat_extracted_t))
    print("norm(after) ", np.linalg.norm(img_calibrated - mat_extracted_t))

    plt.show()
