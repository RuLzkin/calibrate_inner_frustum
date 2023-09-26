"""
The relationship between input RGB and output RGB is:
RGB_output = C_camera * C_led  * RGB_input

The response functions of the LED and Camera are assumed to be fixed.
Therefore, the equation above can be simplified to:
RGB_output = C_system * RGB_input

Here, the system's response function is described by:
C_system = RGB_output * pinv(RGB_input)

Matrix_Calibrate = inv(C_system)

Using this response function, the system can be calibrated:
RGB_input_calibrated = Matrix_Calibrate * RGB_input

RGB_output = C_system * RGB_input_calibrated
 = C_system * inv(C_system) * RGB_input
 = RGB_input
 -> RGB_output = RGB_input

Note: "pinv" refers to the pseudoinverse, and "inv" refers to the inverse.
"""

import numpy as np

# [R, G, B]
#   | 1 0 0 |
# = | 0 1 0 |
#   | 0 0 1 |

# [R, G, B, W]
#   | 1 0 0 1 |
# = | 0 1 0 1 |
#   | 0 0 1 1 |

rgb_input = np.eye(3)
rgb_output = np.eye(3) + 0.1 * np.random.rand(3, 3)

mat_system = np.dot(rgb_output, np.linalg.pinv(rgb_input))

print("LED INPUT RGBs:\n", rgb_input)
print("CAMERA OUTPUT:\n", rgb_output)
print("CALUCULATED C_system:\n", np.dot(mat_system, rgb_input))
print("ERROR (OUTPUT - C_system * INPUT):\n", rgb_output - np.dot(mat_system, rgb_input))
