import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from module_load_exr import load_exr, method_okawa, method_ue, convert_matrix


ndarr_RGB = load_exr("C:/Dropbox/TOEI/20231115_inner_correction/For_inner_Calib_rgbw.0.exr")

# plt.figure()
# plt.imshow(ndarr_RGB)
# plt.show()

pos_r = 1700, 1950, 1600, 1850
pos_g = 1750, 1950, 2350, 2550
pos_b = 1700, 1950, 3000, 3250
pos_w = 1150, 1350, 2300, 2550

r = ndarr_RGB[pos_r[0]:pos_r[1], pos_r[2]:pos_r[3], :].mean(axis=(0, 1))
g = ndarr_RGB[pos_g[0]:pos_g[1], pos_g[2]:pos_g[3], :].mean(axis=(0, 1))
b = ndarr_RGB[pos_b[0]:pos_b[1], pos_b[2]:pos_b[3], :].mean(axis=(0, 1))
w = ndarr_RGB[pos_w[0]:pos_w[1], pos_w[2]:pos_w[3], :].mean(axis=(0, 1))

rgb_input = np.concatenate([np.eye(3), np.ones((3, 1))], axis=1)
rgb_output = np.stack([r, g, b, w], axis=1)

mat_okawa = method_okawa(rgb_input, rgb_output)
mat_ue = method_ue(rgb_input, rgb_output)

ndarr_okawa = convert_matrix(ndarr_RGB, mat_okawa)
ndarr_ue = convert_matrix(ndarr_RGB, mat_ue)

r_okawa = ndarr_okawa[pos_r[0]:pos_r[1], pos_r[2]:pos_r[3], :].mean(axis=(0, 1))
g_okawa = ndarr_okawa[pos_g[0]:pos_g[1], pos_g[2]:pos_g[3], :].mean(axis=(0, 1))
b_okawa = ndarr_okawa[pos_b[0]:pos_b[1], pos_b[2]:pos_b[3], :].mean(axis=(0, 1))
w_okawa = ndarr_okawa[pos_w[0]:pos_w[1], pos_w[2]:pos_w[3], :].mean(axis=(0, 1))

r_ue = ndarr_ue[pos_r[0]:pos_r[1], pos_r[2]:pos_r[3], :].mean(axis=(0, 1))
g_ue = ndarr_ue[pos_g[0]:pos_g[1], pos_g[2]:pos_g[3], :].mean(axis=(0, 1))
b_ue = ndarr_ue[pos_b[0]:pos_b[1], pos_b[2]:pos_b[3], :].mean(axis=(0, 1))
w_ue = ndarr_ue[pos_w[0]:pos_w[1], pos_w[2]:pos_w[3], :].mean(axis=(0, 1))

print(mat_okawa)
print(mat_ue)
# print(np.dot(mat_okawa, np.ones((3, 1))))
# print(np.dot(mat_ue, np.ones((3, 1))))
# print(mat_okawa / np.dot(np.dot(mat_okawa, np.ones((3, 1))), np.ones((1, 3))))
# print(mat_ue / np.dot(np.dot(mat_ue, np.ones((3, 1))), np.ones((1, 3))))
# # print(np.dot(mat_okawa / np.dot(np.dot(mat_okawa, np.ones((3, 1))), np.ones((1, 3))), np.ones((3, 1))))
# # print(np.dot(mat_ue / np.dot(np.dot(mat_ue, np.ones((3, 1))), np.ones((1, 3))), np.ones((3, 1))))
# # print(mat_okawa / mat_okawa.max())
# # print(mat_ue / mat_ue.max())


def make_patches():
    roi_r = pat.Rectangle((pos_r[2], pos_r[0]), height=pos_r[1] - pos_r[0], width=pos_r[3] - pos_r[2], fill=False, ec="red")
    roi_g = pat.Rectangle((pos_g[2], pos_g[0]), height=pos_g[1] - pos_g[0], width=pos_g[3] - pos_g[2], fill=False, ec="green")
    roi_b = pat.Rectangle((pos_b[2], pos_b[0]), height=pos_b[1] - pos_b[0], width=pos_b[3] - pos_b[2], fill=False, ec="blue")
    roi_w = pat.Rectangle((pos_w[2], pos_w[0]), height=pos_w[1] - pos_w[0], width=pos_w[3] - pos_w[2], fill=False, ec="black")
    return roi_r, roi_g, roi_b, roi_w


plt.figure(figsize=(4, 7))
plt.subplot(3, 1, 1)
plt.imshow(ndarr_RGB / ndarr_RGB.max())
# plt.imshow(ndarr_RGB)
plt.xlim(1400, 3500)
plt.ylim(2100, 800)
ax: plt.Axes = plt.gca()
roi_r, roi_g, roi_b, roi_w = make_patches()
# roi_r = pat.Rectangle((pos_r[2], pos_r[0]), height=pos_r[1] - pos_r[0], width=pos_r[3] - pos_r[2], fill=False, ec="red")
# roi_g = pat.Rectangle((pos_g[2], pos_g[0]), height=pos_g[1] - pos_g[0], width=pos_g[3] - pos_g[2], fill=False, ec="green")
# roi_b = pat.Rectangle((pos_b[2], pos_b[0]), height=pos_b[1] - pos_b[0], width=pos_b[3] - pos_b[2], fill=False, ec="blue")
# roi_w = pat.Rectangle((pos_w[2], pos_w[0]), height=pos_w[1] - pos_w[0], width=pos_w[3] - pos_w[2], fill=False, ec="black")
ax.add_patch(roi_r)
ax.add_patch(roi_g)
ax.add_patch(roi_b)
ax.add_patch(roi_w)
plt.title('Input')

plt.subplot(3, 1, 2)
plt.imshow(ndarr_okawa / ndarr_okawa.max())
# plt.imshow(ndarr_okawa)
plt.xlim(1400, 3500)
plt.ylim(2100, 800)
ax: plt.Axes = plt.gca()
roi_r, roi_g, roi_b, roi_w = make_patches()
# roi_r = pat.Rectangle((1700, 1400), height=1600 - 1400, width=1900 - 1700, fill=False, ec="red")
# roi_g = pat.Rectangle((2200, 1400), height=1600 - 1400, width=2400 - 2200, fill=False, ec="green")
# roi_b = pat.Rectangle((2600, 1400), height=1600 - 1400, width=2800 - 2600, fill=False, ec="blue")
# roi_w = pat.Rectangle((2200, 900), height=1100 - 900, width=2400 - 2200, fill=False, ec="black")
ax.add_patch(roi_r)
ax.add_patch(roi_g)
ax.add_patch(roi_b)
ax.add_patch(roi_w)
plt.title('okawa')

plt.subplot(3, 1, 3)
plt.imshow(ndarr_ue / ndarr_ue.max())
# plt.imshow(ndarr_ue)
plt.xlim(1400, 3500)
plt.ylim(2100, 800)
ax: plt.Axes = plt.gca()
roi_r, roi_g, roi_b, roi_w = make_patches()
# roi_r = pat.Rectangle((1700, 1400), height=1600 - 1400, width=1900 - 1700, fill=False, ec="red")
# roi_g = pat.Rectangle((2200, 1400), height=1600 - 1400, width=2400 - 2200, fill=False, ec="green")
# roi_b = pat.Rectangle((2600, 1400), height=1600 - 1400, width=2800 - 2600, fill=False, ec="blue")
# roi_w = pat.Rectangle((2200, 900), height=1100 - 900, width=2400 - 2200, fill=False, ec="black")
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
