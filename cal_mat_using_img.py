import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as pch
import cv2
from tqdm import trange


def convert_matrix(img, matrix):
    out = np.zeros(img.shape)
    for _i in trange(img.shape[0]):
        for _j in range(img.shape[1]):
            out[_i, _j, :] = np.dot(matrix, img[_i, _j, :])
    return out


def method_okawa(input, output):
    return np.linalg.inv(np.dot(output, np.linalg.pinv(input)))


def method_ue(input, output):
    W = np.sum(output, axis=1)
    Imat = np.linalg.inv(output)
    Wnor = W / W.max()
    S = np.dot(Imat, Wnor)
    Smat = np.diag(S)
    C = np.dot(Smat, output)
    return np.linalg.inv(C)


img_a = (cv2.imread("./testfiles/simple_config.png")[..., ::-1] / 255) ** (1 / 1)
img_b = (cv2.imread("./testfiles/test_config.png")[..., ::-1] / 255) ** (1 / 1)
# img_a = (cv2.imread("./testfiles/simple_config.png")[..., ::-1] / 255) ** (1 / 2.35)
# img_b = (cv2.imread("./testfiles/test_config.png")[..., ::-1] / 255) ** (1 / 2.35)

mat_a = np.eye(4)
mat_b = np.reshape(np.array([10, 5, -20, 0, 20, 4, 9, 0, 2, -4, 8, 0, 0, 0, 0, 1]), (4, 4))

img_a2b = convert_matrix(img_a, mat_b[:3, :3])
img_b2a = convert_matrix(img_b, np.linalg.pinv(mat_b[:3, :3]))

print(np.linalg.pinv(mat_b[:3, :3]))

img_a2b /= img_a2b.max()
img_b2a /= img_b2a.max()

plt.figure(figsize=(7, 5))
ax1 = plt.subplot(2, 2, 1)
plt.imshow(img_a)
plt.title("input (led)")
ax2 = plt.subplot(2, 2, 2)
plt.imshow(img_b)
plt.title("output (camera)")
plt.subplot(2, 2, 3)
plt.imshow(img_a2b)
plt.title("calculated (mat * input)")
plt.subplot(2, 2, 4)
plt.imshow(img_b2a)
plt.title("calculated ($mat^{-1}$ * output)")
plt.tight_layout()


r = pch.Rectangle(xy=(183, 677), width=30, height=30, ec='red', fill=False)
g = pch.Rectangle(xy=(520, 659), width=30, height=30, ec='red', fill=False)
b = pch.Rectangle(xy=(690, 690), width=30, height=30, ec='red', fill=False)
w = pch.Rectangle(xy=(970, 450), width=30, height=30, ec='red', fill=False)
ax1.add_patch(r)
ax1.add_patch(g)
ax1.add_patch(b)
ax1.add_patch(w)
r = pch.Rectangle(xy=(183, 677), width=30, height=30, ec='red', fill=False)
g = pch.Rectangle(xy=(520, 659), width=30, height=30, ec='red', fill=False)
b = pch.Rectangle(xy=(690, 690), width=30, height=30, ec='red', fill=False)
w = pch.Rectangle(xy=(970, 450), width=30, height=30, ec='red', fill=False)
ax2.add_patch(r)
ax2.add_patch(g)
ax2.add_patch(b)
ax2.add_patch(w)

rgb_input = np.stack([
    img_a[677:707, 183:213, :].mean(axis=(0, 1)),
    img_a[659:689, 520:550, :].mean(axis=(0, 1)),
    img_a[690:720, 690:720, :].mean(axis=(0, 1)),
], axis=1)
rgb_output = np.stack([
    img_b[677:707, 183:213, :].mean(axis=(0, 1)),
    img_b[659:689, 520:550, :].mean(axis=(0, 1)),
    img_b[690:720, 690:720, :].mean(axis=(0, 1)),
], axis=1)

# print(np.dot(mat_b[:3, :3], rgb_input))
# print(mat_b[:3, :3])
# print(np.linalg.pinv(mat_b[:3, :3]))

mat_okawa = method_okawa(rgb_input, rgb_output)
mat_ue = method_ue(rgb_input, rgb_output)

# img_okawa = convert_matrix(img_a, np.linalg.inv(mat_okawa))
# img_ue = convert_matrix(img_a, np.linalg.inv(mat_ue))

# gamma calib okawa
# [[1.18050323 1.12930092 0.        ]
#  [1.22887017 1.1124652  1.17263147]
#  [1.05673298 0.         1.16467454]]
# gamma calib ue
# [[ 3.46855166 -3.52104369  3.54509907]
#  [-0.51419639  3.68068723 -3.70583327]
#  [-3.14712216  3.19474978 -0.19943052]]
# no gamma calib okawa
# [[1.47692308 1.33076923 0.        ]
#  [1.62307692 1.28461538 1.45388343]
#  [1.13846154 0.         1.43080592]]
# no gamma calib ue
# [[ 4.38822604 -4.54588686  4.61920761]
#  [-1.5927133   5.04514611 -5.12651943]
#  [-3.49170674  3.61715729 -0.62711144]]

# plt.figure()
# ax1 = plt.subplot(2, 1, 1)
# plt.imshow(img_okawa)
# ax2 = plt.subplot(2, 1, 2)
# plt.imshow(img_ue)

plt.show()
