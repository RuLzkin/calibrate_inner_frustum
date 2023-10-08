import array
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from tqdm import trange


def load_exr(path):
    exr = OpenEXR.InputFile(path)

    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    img_R, img_G, img_B = exr.channels('RGB', pt)

    arr_R = array.array('f', img_R)
    arr_G = array.array('f', img_G)
    arr_B = array.array('f', img_B)

    ndarr_R = np.array(arr_R, dtype="float32")
    ndarr_G = np.array(arr_G, dtype="float32")
    ndarr_B = np.array(arr_B, dtype="float32")

    # list_buff = []
    # for _i in trange(len(ndarr_R)):
    #     list_buff.append([ndarr_R[_i], ndarr_G[_i], ndarr_B[_i]])
    # # for r, g, b in tqdm(zip(ndarr_R, ndarr_G, ndarr_B)):
    # #     list_buff.append([r, g, b])

    # ndarr_RGB = np.array(list_buff, dtype="float32")
    ndarr_RGB = np.stack([ndarr_R, ndarr_G, ndarr_B], axis=1)

    # ndarr_RGB = np.array([[r, g, b] for r, g, b in zip(ndarr_R, ndarr_G, ndarr_B)], dtype="float32")
    ndarr_RGB = ndarr_RGB.reshape(size[1], size[0], 3)

    return ndarr_RGB


def convert_matrix(img, matrix):
    out = np.zeros(img.shape)
    for _i in trange(img.shape[0]):
        for _j in range(img.shape[1]):
            out[_i, _j, :] = np.dot(matrix, img[_i, _j, :])
    return out


def method_okawa(input, output):
    output /= output.max()
    return np.linalg.inv(np.dot(output, np.linalg.pinv(input)))


def method_ue(input, output):
    W = output[:, 3]
    output = output[:, :3]
    Imat = np.linalg.inv(output)
    Wnor = W / W.max()
    S = np.dot(Imat, Wnor)
    Smat = np.diag(S)
    C = np.dot(Smat, output)
    return np.linalg.inv(C)


ndarr_RGB = load_exr("./testfiles/A_0006C049_230928_184717_a1CB4.90.exr")
ndarr_okawa = load_exr("./testfiles/okawamatrix.exr")
ndarr_ue = load_exr("./testfiles/uematrix.exr")
# ndarr_RGB = load_exr("./testfiles/okawamatrix.exr")

# print(ndarr_RGB.shape)
# print(ndarr_RGB.max())
# print(ndarr_RGB.min())

# ndarr_RGB[900:1000, 2300:2400, 0] = 1

r = ndarr_RGB[1400:1700, 1616:1911, :].mean(axis=(0, 1))
g = ndarr_RGB[1400:1700, 2200:2500, :].mean(axis=(0, 1))
b = ndarr_RGB[1400:1700, 2800:3100, :].mean(axis=(0, 1))
w = ndarr_RGB[800:1100, 2200:2500, :].mean(axis=(0, 1))

r_okawa = ndarr_okawa[1400:1600, 1700:1900, :].mean(axis=(0, 1))
g_okawa = ndarr_okawa[1400:1600, 2200:2400, :].mean(axis=(0, 1))
b_okawa = ndarr_okawa[1400:1600, 2600:2800, :].mean(axis=(0, 1))
w_okawa = ndarr_okawa[900:1100, 2200:2400, :].mean(axis=(0, 1))

r_ue = ndarr_ue[1400:1600, 1700:1900, :].mean(axis=(0, 1))
g_ue = ndarr_ue[1400:1600, 2200:2400, :].mean(axis=(0, 1))
b_ue = ndarr_ue[1400:1600, 2600:2800, :].mean(axis=(0, 1))
w_ue = ndarr_ue[900:1100, 2200:2400, :].mean(axis=(0, 1))

print(r, g, b, w)

rgb_input = np.concatenate([np.eye(3), np.ones((3, 1))], axis=1)

rgb_output = np.stack([r, g, b, w], axis=1)

mat_okawa = method_okawa(rgb_input, rgb_output)
mat_ue = method_ue(rgb_input, rgb_output)

print(mat_okawa)
print(mat_ue)
print(np.dot(mat_okawa, np.ones((3, 1))))
print(np.dot(mat_ue, np.ones((3, 1))))
print(mat_okawa / np.dot(np.dot(mat_okawa, np.ones((3, 1))), np.ones((1, 3))))
print(mat_ue / np.dot(np.dot(mat_ue, np.ones((3, 1))), np.ones((1, 3))))
# print(np.dot(mat_okawa / np.dot(np.dot(mat_okawa, np.ones((3, 1))), np.ones((1, 3))), np.ones((3, 1))))
# print(np.dot(mat_ue / np.dot(np.dot(mat_ue, np.ones((3, 1))), np.ones((1, 3))), np.ones((3, 1))))
# print(mat_okawa / mat_okawa.max())
# print(mat_ue / mat_ue.max())

plt.figure(figsize=(4, 7))
plt.subplot(3, 1, 1)
plt.imshow(ndarr_RGB / ndarr_RGB.max())
# plt.imshow(ndarr_RGB)
ax: plt.Axes = plt.gca()
roi_r = pat.Rectangle((1616, 1400), height=1700 - 1400, width=1911 - 1616, fill=False, ec="red")
roi_g = pat.Rectangle((2200, 1400), height=1700 - 1400, width=2500 - 2200, fill=False, ec="green")
roi_b = pat.Rectangle((2800, 1400), height=1700 - 1400, width=3100 - 2800, fill=False, ec="blue")
roi_w = pat.Rectangle((2200, 800), height=1100 - 800, width=2500 - 2200, fill=False, ec="black")
ax.add_patch(roi_r)
ax.add_patch(roi_g)
ax.add_patch(roi_b)
ax.add_patch(roi_w)
plt.title('Input')

plt.subplot(3, 1, 2)
plt.imshow(ndarr_okawa / ndarr_okawa.max())
# plt.imshow(ndarr_okawa)
ax: plt.Axes = plt.gca()
roi_r = pat.Rectangle((1700, 1400), height=1600 - 1400, width=1900 - 1700, fill=False, ec="red")
roi_g = pat.Rectangle((2200, 1400), height=1600 - 1400, width=2400 - 2200, fill=False, ec="green")
roi_b = pat.Rectangle((2600, 1400), height=1600 - 1400, width=2800 - 2600, fill=False, ec="blue")
roi_w = pat.Rectangle((2200, 900), height=1100 - 900, width=2400 - 2200, fill=False, ec="black")
ax.add_patch(roi_r)
ax.add_patch(roi_g)
ax.add_patch(roi_b)
ax.add_patch(roi_w)
plt.title('okawa')

plt.subplot(3, 1, 3)
plt.imshow(ndarr_ue / ndarr_ue.max())
# plt.imshow(ndarr_ue)
ax: plt.Axes = plt.gca()
roi_r = pat.Rectangle((1700, 1400), height=1600 - 1400, width=1900 - 1700, fill=False, ec="red")
roi_g = pat.Rectangle((2200, 1400), height=1600 - 1400, width=2400 - 2200, fill=False, ec="green")
roi_b = pat.Rectangle((2600, 1400), height=1600 - 1400, width=2800 - 2600, fill=False, ec="blue")
roi_w = pat.Rectangle((2200, 900), height=1100 - 900, width=2400 - 2200, fill=False, ec="black")
ax.add_patch(roi_r)
ax.add_patch(roi_g)
ax.add_patch(roi_b)
ax.add_patch(roi_w)
plt.title('ue')

plt.tight_layout()

x_plot = np.array([1, 2, 3])

plt.figure(figsize=(6, 7))
ax = plt.subplot(3, 1, 1)
plt.bar(x_plot - 0.3, r, 0.2, label="red patch", color="red")
plt.bar(x_plot - 0.1, g, 0.2, label="green patch", color="green")
plt.bar(x_plot + 0.1, b, 0.2, label="blue patch", color="blue")
plt.bar(x_plot + 0.3, w, 0.2, label="white patch", color="white", ec="black")
plt.ylabel("Not calibrated")
# plt.xlabel("RGB Value")
plt.xticks(x_plot, ["R", "G", "B"])
plt.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = plt.subplot(3, 1, 2)
plt.bar(x_plot - 0.3, r_okawa, 0.2, label="red patch", color="red")
plt.bar(x_plot - 0.1, g_okawa, 0.2, label="green patch", color="green")
plt.bar(x_plot + 0.1, b_okawa, 0.2, label="blue patch", color="blue")
plt.bar(x_plot + 0.3, w_okawa, 0.2, label="white patch", color="white", ec="black")
plt.ylabel("calibrated (okawa)")
# plt.xlabel("RGB Value")
plt.xticks(x_plot, ["R", "G", "B"])
plt.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = plt.subplot(3, 1, 3)
plt.bar(x_plot - 0.3, r_ue, 0.2, label="red patch", color="red")
plt.bar(x_plot - 0.1, g_ue, 0.2, label="green patch", color="green")
plt.bar(x_plot + 0.1, b_ue, 0.2, label="blue patch", color="blue")
plt.bar(x_plot + 0.3, w_ue, 0.2, label="white patch", color="white", ec="black")
plt.ylabel("Not calibrated (ue)")
plt.xlabel("RGB Value")
plt.xticks(x_plot, ["R", "G", "B"])
plt.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.suptitle("Real Value")
plt.tight_layout()


plt.figure(figsize=(6, 7))
ax = plt.subplot(3, 1, 1)
plt.bar(x_plot - 0.3, r / r.max(), 0.2, label="red patch", color="red")
plt.bar(x_plot - 0.1, g / g.max(), 0.2, label="green patch", color="green")
plt.bar(x_plot + 0.1, b / b.max(), 0.2, label="blue patch", color="blue")
plt.bar(x_plot + 0.3, w / w.max(), 0.2, label="white patch", color="white", ec="black")
plt.ylabel("Not calibrated")
# plt.xlabel("RGB Value")
plt.xticks(x_plot, ["R", "G", "B"])
plt.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = plt.subplot(3, 1, 2)
plt.bar(x_plot - 0.3, r_okawa / r_okawa.max(), 0.2, label="red patch", color="red")
plt.bar(x_plot - 0.1, g_okawa / g_okawa.max(), 0.2, label="green patch", color="green")
plt.bar(x_plot + 0.1, b_okawa / b_okawa.max(), 0.2, label="blue patch", color="blue")
plt.bar(x_plot + 0.3, w_okawa / w_okawa.max(), 0.2, label="white patch", color="white", ec="black")
plt.ylabel("calibrated (okawa)")
# plt.xlabel("RGB Value")
plt.xticks(x_plot, ["R", "G", "B"])
plt.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = plt.subplot(3, 1, 3)
plt.bar(x_plot - 0.3, r_ue / r_ue.max(), 0.2, label="red patch", color="red")
plt.bar(x_plot - 0.1, g_ue / g_ue.max(), 0.2, label="green patch", color="green")
plt.bar(x_plot + 0.1, b_ue / b_ue.max(), 0.2, label="blue patch", color="blue")
plt.bar(x_plot + 0.3, w_ue / w_ue.max(), 0.2, label="white patch", color="white", ec="black")
plt.ylabel("Not calibrated (ue)")
plt.xlabel("RGB Value")
plt.xticks(x_plot, ["R", "G", "B"])
plt.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.suptitle("Normalized Value")
plt.tight_layout()

plt.show()
