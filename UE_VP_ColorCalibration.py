import numpy as np

R = np.array([0.55188, 0.02992, 0.04264])
G = np.array([0.24531, 0.75189, 0.06259])
B = np.array([0.01521, 0.01602, 0.53038])
W = np.array([0.76859, 0.72502, 0.71202])

Fmat = np.stack([R, G, B], axis=1)
# print(Fmat)

Imat = np.linalg.inv(Fmat)
# print(Imat)

Wnor = W / W.max()
# print(Wnor)

S = np.dot(Imat, Wnor)
Smat = np.diag(S)
# print(Smat)

C = np.dot(Smat, Fmat)
CalibrationMX = np.linalg.inv(C)
print(C)
print(CalibrationMX)
# [[ 1.47877064 -0.51141253 -0.02311738]
#  [-0.05645372  1.15644992 -0.02582645]
#  [-0.11222396 -0.09535723  1.25532589]]
