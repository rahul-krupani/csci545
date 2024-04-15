import copy

import matplotlib.pyplot as plt
import numpy as np


class KalmanFilter():
    """
    Implementation of a Kalman Filter.
    """
    def __init__(self, mu, sigma, A, C, R=0., Q=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param A: process model
        :param C: measurement model
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.A = A
        self.R = R
        # measurement model
        self.C = C
        self.Q = Q

    def reset(self):
        """
        Reset belief state to initial value.
        """
        self.mu = self.mu_init
        self.sigma = self.sigma_init

    def run(self, sensor_data):
        """
        Run the Kalman Filter using the given sensor updates.

        :param sensor_data: array of T sensor updates as a TxS array.

        :returns: A tuple of predicted means (as a TxD array) and predicted
                  covariances (as a TxDxD array) representing the KF's belief
                  state AFTER each predict/update cycle, over T timesteps.
        """
        means = []
        covs = []
        for i in range(sensor_data.shape[0]):
            
            mut, sigmat, K = self._predict()
            self._update(sensor_data[i], mut, sigmat, K)
            
            means.append(self.mu[:, 0])
            covs.append(self.sigma)
            
        return np.array(means), np.array(covs)

    def _predict(self):
        mut = np.matmul(self.A, self.mu)
        Sigmat = np.matmul(np.matmul(self.A, self.sigma), self.A.T) + self.R
        K = np.matmul(np.matmul(Sigmat, self.C.T), np.linalg.inv(np.matmul(np.matmul(self.C, Sigmat), self.C.T) + self.Q))
        return mut, Sigmat, K

    def _update(self, z, mut, Sigmat, K):
        self.mu = mut + np.matmul(K, z - np.matmul(self.C, mut))
        self.sigma = np.matmul(np.identity(4) - np.matmul(K, self.C), Sigmat)

def plot_prediction(t, ground_truth, measurement, predict_mean, predict_cov):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param measurement: Tx1 array of sensor values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    predict_pos_mean = predict_mean[:, 0]
    predict_pos_std = predict_cov[:, 0, 0]

    plt.figure()
    plt.plot(t, ground_truth, color='k')
    plt.plot(t, measurement, color='r')
    plt.plot(t, predict_pos_mean, color='g')
    plt.fill_between(
        t,
        predict_pos_mean-predict_pos_std,
        predict_pos_mean+predict_pos_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground truth", "measurements", "predictions"))
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("Predicted Values")
    plt.savefig('problem2a_kf_estimation.png')
    plt.show()


def plot_mse(t, ground_truth, predict_means, first):
    """
    Plot MSE of your KF over many trials.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_means: NxTxD array of T mean vectors over N trials
    """
    predict_pos_means = predict_means[:, :, 0]
    errors = ground_truth.squeeze() - predict_pos_means
    mse = np.mean(errors, axis=0) ** 2

    plt.figure()
    plt.plot(t, mse)
    plt.xlabel("time (s)")
    plt.ylabel("position MSE (m^2)")
    plt.title("Prediction Mean-Squared Error")
    if first:
        plt.savefig('problem2a_kf_mse.png')
    else:
        plt.savefig('problem2b_kf_mse.png')
    plt.show()


def problem2a():
    kalman = KalmanFilter(mu=np.array([[5., 1., 0., 0.]]).T,
                          sigma=np.array([[10., 0., 0., 0.],
                                          [0., 10., 0., 0.],
                                          [0., 0., 10., 0.],
                                          [0., 0., 0., 10.]]),
                          A=np.array([[1., 0.1, 0., 0.],
                                      [0., 1., 0.1, 0.],
                                      [0., 0., 1., 0.1],
                                      [0., 0., 0., 1.]]),
                          C=np.array([[1., 0., 0., 0.]]),
                          R=0.,
                          Q=1.0)

    T = 100
    p = 0
    dt = 0.1

    ground_truths = []
    t = []
    for i in range(T):
        t.append((i+1)*dt)
        ground_truths.append(np.array([np.sin(0.1*(i+1))]))
    ground_truths = np.array(ground_truths)
    t = np.array(t)

    N = 10000
    mse_predict_means = []
    for i in range(N):
        kalman.reset()

        sensor_data = []
        for j in range(T):
            sensor_data.append(np.array([ground_truths[j][0]+np.random.normal(0.0, 1.0)]))

        predict_means, predict_covariances = kalman.run(np.array(sensor_data))
        mse_predict_means.append(copy.deepcopy(predict_means))

    mse_predict_means = np.array(mse_predict_means)
    plot_mse(t, ground_truths, mse_predict_means, True)
    plot_prediction(t, ground_truths, sensor_data, predict_means, predict_covariances)




def problem2b():
    kalman = KalmanFilter(mu=np.array([[5., 1., 0., 0.]]).T,
                          sigma=np.array([[10., 0., 0., 0.],
                                          [0., 10., 0., 0.],
                                          [0., 0., 10., 0.],
                                          [0., 0., 0., 10.]]),
                          A=np.array([[1., 0.1, 0., 0.],
                                      [0., 1., 0.1, 0.],
                                      [0., 0., 1., 0.1],
                                      [0., 0., 0., 1.]]),
                          C=np.array([[1., 0., 0., 0.]]),
                          R=np.array([[0.1, 0, 0., 0.],
                                      [0., 0.1, 0, 0.],
                                      [0., 0., 0.1, 0],
                                      [0., 0., 0., 0.1]]),
                          Q=1.0)

    T = 100
    p = 0
    dt = 0.1

    ground_truths = []
    t = []
    for i in range(T):
        t.append((i + 1) * dt)
        ground_truths.append(np.array([np.sin(0.1 * (i+1))]))
    ground_truths = np.array(ground_truths)
    t = np.array(t)

    N = 10000
    mse_predict_means = []
    for i in range(N):
        kalman.reset()

        sensor_data = []
        for j in range(T):
            sensor_data.append(np.array([ground_truths[j][0] + np.random.normal(0.0, 1.0)]))

        predict_means, predict_covariances = kalman.run(np.array(sensor_data))
        mse_predict_means.append(copy.deepcopy(predict_means))

    mse_predict_means = np.array(mse_predict_means)
    plot_mse(t, ground_truths, mse_predict_means, False)

if __name__ == '__main__':
    problem2a()
    problem2b()
