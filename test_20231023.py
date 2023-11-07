import numpy as np
from calculate_from_exr import method_ue, method_okawa


# 20230929
# r = np.array([0.02506, 0.00758, 0.00106])
# g = np.array([0.01524, 0.03177, 0.00208])
# b = np.array([0.00852, 0.00570, 0.02583])
# w = np.array([0.03719, 0.04031, 0.03003])

# 20231023
r = np.array([0.61136, 0.09565, 0.00961])
g = np.array([0.23019, 0.59589, 0.03279])
b = np.array([0.11515, 0.05697, 0.58426])
w = np.array([0.61053, 0.63168, 0.62524])

rgb_input = np.concatenate([np.eye(3), np.ones((3, 1))], axis=1)
rgb_output = np.stack([r, g, b, w], axis=1)

print(method_ue(None, rgb_output))
print(method_okawa(rgb_input, rgb_output))
