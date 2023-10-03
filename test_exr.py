import array
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


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


img = OpenEXR.InputFile("./testfiles/A_0006C049_230928_184717_a1CB4.90.exr")

dw = img.header()['dataWindow']
size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

pt = Imath.PixelType(Imath.PixelType.FLOAT)

img_R, img_G, img_B = img.channels('RGB', pt)

arr_R = array.array('f', img_R)
arr_G = array.array('f', img_G)
arr_B = array.array('f', img_B)

ndarr_R = np.array(arr_R, dtype="float32")
ndarr_G = np.array(arr_G, dtype="float32")
ndarr_B = np.array(arr_B, dtype="float32")

list_buff = []
for _i in trange(len(ndarr_R)):
    list_buff.append([ndarr_R[_i], ndarr_G[_i], ndarr_B[_i]])
# for r, g, b in tqdm(zip(ndarr_R, ndarr_G, ndarr_B)):
#     list_buff.append([r, g, b])

ndarr_RGB = np.array(list_buff, dtype="float32")
# ndarr_RGB = np.array([[r, g, b] for r, g, b in zip(ndarr_R, ndarr_G, ndarr_B)], dtype="float32")
ndarr_RGB = ndarr_RGB.reshape(size[1], size[0], 3)

print(ndarr_RGB.shape)
print(ndarr_RGB.max())
print(ndarr_RGB.min())

# ndarr_RGB[900:1000, 2300:2400, 0] = 1

r = ndarr_RGB[1400:1700, 1616:1911, :].mean(axis=(0, 1))
g = ndarr_RGB[1400:1700, 2200:2500, :].mean(axis=(0, 1))
b = ndarr_RGB[1400:1700, 2800:3100, :].mean(axis=(0, 1))
w = ndarr_RGB[800:1100, 2200:2500, :].mean(axis=(0, 1))

print(r, g, b, w)

rgb_input = np.concatenate([np.eye(3), np.ones((3, 1))], axis=1)

rgb_output = np.stack([r, g, b, w], axis=1)

mat_okawa = method_okawa(rgb_input, rgb_output)
mat_ue = method_ue(rgb_input, rgb_output)

print(mat_okawa)
print(mat_ue)
print(mat_okawa / mat_okawa.max())
print(mat_ue / mat_ue.max())

plt.figure()
plt.imshow(ndarr_RGB / ndarr_RGB.max())
plt.title('Input')
plt.show()
