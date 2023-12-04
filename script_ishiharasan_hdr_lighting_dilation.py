import cv2
import numpy as np
from matplotlib import pyplot as plt
from module_load_exr import preview_exr, load_exr


def dilate_saturated_pixels(hdri_map, max_channel_value=1.0):
    # Step 1: Identify saturated pixels
    saturated_pixels = np.any(hdri_map > max_channel_value, axis=-1)

    # Step 2: Group saturated pixels into connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(saturated_pixels.astype(np.uint8))

    # Step 3: Process connected components
    for label in range(1, len(stats)):
        cc_mask = (labels == label)

        # Step 3(a): Compute the CC’s average pixel value
        cc_average = np.mean(hdri_map[cc_mask], axis=0)

        # Step 3(b): If no channel of 𝑥¯ exceeds 1, reset the entire CC’s pixel values with 𝑥¯.
        if np.all(cc_average <= max_channel_value):
            hdri_map[cc_mask] = cc_average
        else:
            # Step 3(c): Dilate the perimeter of the CC by one pixel
            dilated_mask = cv2.dilate(cc_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)

            # Avoid intersections with other CC’s
            dilated_mask = np.logical_and(dilated_mask, np.logical_not(saturated_pixels))

            # Step 3(d): If any CC’s were dilated, repeat step 3
            if np.any(dilated_mask):
                hdri_map[dilated_mask] = cc_average
                return dilate_saturated_pixels(hdri_map, max_channel_value)

    return hdri_map


hdri_input = load_exr(r"C:\Dropbox\TOEI\20231204_LightingDilation\test.exr")

# hdri_input[hdri_input > 1.0] = 1.0

result = dilate_saturated_pixels(hdri_input.copy(), 1.0)

print(hdri_input.shape)
print(hdri_input.dtype)
print(hdri_input.max())

plt.figure()
for _i in range(3):
    plt.subplot(3, 1, _i + 1)
    plt.hist(hdri_input[..., _i].ravel(), bins=100, range=(0, 2))

plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(hdri_input)
plt.subplot(2, 1, 2)
plt.imshow(result)
plt.show()
# cv2.imwrite('image/result.hdr', result)
