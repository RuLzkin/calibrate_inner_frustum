import numpy as np
from matplotlib import pyplot as plt
import cv2


img_a = cv2.imread("./testfiles/simple_config.png")[..., ::-1]
img_b = cv2.imread("./testfiles/test_config.png")[..., ::-1]

mat_a = np.eye(4)
mat_b = np.reshape(np.array([10, 5, -20, 0, 20, 4, 9, 0, 2, -4, 8, 0, 0, 0, 0, 1]), (4, 4))

plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(img_a)
plt.subplot(2, 1, 2)
plt.imshow(img_b)

plt.show()
