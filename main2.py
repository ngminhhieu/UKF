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
  F = np.array([[1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1]], dtype=float)
  return np.dot(F, x)

def hx(x):
  # measurement function - convert state into a measurement
  # where measurements are [x_pos, y_pos]
  H = np.array([[1, 1],
                [0, 0],
                [1, 1],
                [0, 0]], dtype=float)
  return np.dot(x, H)
  # return np.array([x[0], x[2]])

dt = 0.1
# create sigma points to use in the filter. This is standard for Gaussian processes
points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)

print("points.wm shape: ", points.Wm.shape)
print("points.wc shape: ", points.Wc.shape)

kf = UKF(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)
kf.x = np.array([-1., 1., -1., 1]) # initial state
kf.P *= 0.2 # initial uncertainty
z_std = 0.1
kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)

print("P shape: ", kf.P.shape)
print("R shape: ", kf.R.shape)
print("Q shape: ", kf.Q.shape)

zs = [[i+randn()*z_std, i+randn()*z_std] for i in range(50)] # measurements
for z in zs:
    # print("sigma_f shape: ", kf.sigmas_f.shape)
    # print("x shape: ", kf.x.shape)
    kf.predict()
    # print("sigma_h shape: ", kf.sigmas_h.shape)
    kf.update(z)
    # import sys
    # sys.exit()
    print(kf.x, 'log-likelihood', z)


# data_npz = 'drive/My Drive/map_gauge_72_stations.npz'
# measurement_precip = np.load(data_npz)['map_precip']
# actual_precip = np.load(data_npz)['gauge_precip']

# R: error in measurement
# Q: Process noise covariance matrix (trong estimation)
# P: error in estimation