from kalman.unscented_transform import unscented_transform
from kalman.sigma_points import MerweScaledSigmaPoints
from kalman.UKF import UnscentedKalmanFilter as UKF
from common.discretization import Q_discrete_white_noise
import numpy as np
from numpy.random import randn
from pandas import read_csv

def fx(x, dt):
  # state transition function - predict next state based
  # on constant velocity model x = vt + x_0
  F = np.array([[1, dt],
                [0, 1]], dtype=float)
  return np.dot(F, x)

def hx(x):
  # measurement function - convert state into a measurement
  # where measurements are [x_pos, y_pos]
  return np.array(x[0])

dt = 0.1
# create sigma points to use in the filter. This is standard for Gaussian processes
points = MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=-1)

print("points.wm shape: ", points.Wm.shape)
print("points.wc shape: ", points.Wc.shape)

# z - measurement chỉ đo được vị trí nên chiều là 1
# x - estimation, ước lượng vị trí và tốc độ di chuyển theo phương trình toán học, nên chiều là 2
kf = UKF(dim_x=2, dim_z=1, dt=dt, fx=fx, hx=hx, points=points)
kf.x = np.array([-1., 1.]) # initial state
kf.P *= 0.2 # initial uncertainty
z_std = 0.1
kf.R = np.diag([z_std**2]) # do z có dim là 1
kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=1)

print("P shape: ", kf.P.shape)
print("R shape: ", kf.R.shape)
print("Q shape: ", kf.Q.shape)

zs = [i+randn()*z_std for i in range(50)] # measurements
for z in zs:
    print("sigma_f shape: ", kf.sigmas_f.shape)
    print("x shape: ", kf.x.shape)
    kf.predict()
    print("sigma_h shape: ", kf.sigmas_h.shape)
    kf.update(z)
    print("Estimation (adjustment): ", kf.x)
    print("Actual (gauge): ", 1)
    print("Measurement (map): ", z)