import matplotlib.pyplot as plt
import numpy as np


class ExtendedKalmanFilter():
    """
    Implementation of an Extended Kalman Filter.
    """

    def __init__(self, mu, sigma, R=0., Q=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param g: process function
        :param g_jac: process function's jacobian
        :param h: measurement function
        :param h_jac: measurement function's jacobian
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.g = self.get_g()
        self.g_jac = self.get_g_jac()
        self.R = R
        # measurement model
        self.h = self.get_h()
        self.h_jac = self.get_h_jac()
        self.Q = Q

    def reset(self):
        """
        Reset belief state to initial value.
        """
        self.mu = self.mu_init
        self.sigma = self.sigma_init

    def get_g(self):
        return np.array([[self.mu[1][0] * self.mu[0][0], self.mu[1][0]]]).T

    def get_g_jac(self):
        return np.array([[self.mu[1][0], self.mu[0][0]], [0., 1.]])\

    def get_h(self):
        return np.array([np.sqrt((self.mu[0][0] ** 2) + 1)])

    def get_h_jac(self):
        return np.array([[self.mu[0][0] / np.sqrt((self.mu[0][0] ** 2) + 1), 0]])

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
            
            self.g = self.get_g()
            self.g_jac = self.get_g_jac()
            self.h = self.get_h()
            self.h_jac = self.get_h_jac()

            
            mut, sigmat, K = self._predict()
            self._update(sensor_data[i], mut, sigmat, K)
            
            means.append(self.mu[:, 0])
            covs.append(self.sigma)
            
        return np.array(means), np.array(covs)

    def _predict(self):
        mut = self.g
        Sigmat = np.matmul(np.matmul(self.g_jac, self.sigma), self.g_jac.T) + self.R
        K = np.matmul(np.matmul(Sigmat, self.h_jac.T),
                      np.linalg.inv(np.matmul(np.matmul(self.h_jac, Sigmat), self.h_jac.T) + self.Q))
        return mut, Sigmat, K

    def _update(self, z, mut, Sigmat, K):
        self.mu = mut + np.matmul(K, np.expand_dims(z - self.h, axis=1))
        self.sigma = np.matmul(np.identity(2) - np.matmul(K, self.h_jac), Sigmat)


def plot_prediction(t, ground_truth, predict_mean, predict_cov):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    gt_x, gt_a = ground_truth[:, 0], ground_truth[:, 1]
    pred_x, pred_a = predict_mean[:, 0], predict_mean[:, 1]
    pred_x_std = np.sqrt(predict_cov[:, 0, 0])
    pred_a_std = np.sqrt(predict_cov[:, 1, 1])

    plt.figure(figsize=(7, 10))
    plt.subplot(211)
    plt.plot(t, gt_x, color='k')
    plt.plot(t, pred_x, color='g')
    plt.fill_between(
        t,
        pred_x - pred_x_std,
        pred_x + pred_x_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$x$")
    plt.title(r"EKF estimation: $x$")

    plt.subplot(212)
    plt.plot(t, gt_a, color='k')
    plt.plot(t, pred_a, color='g')
    plt.fill_between(
        t,
        pred_a - pred_a_std,
        pred_a + pred_a_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$\alpha$")
    plt.title(r"EKF estimation: $\alpha$")

    plt.savefig('problem3_ekf_estimation.png')
    plt.show()


def problem3():
    kalman = ExtendedKalmanFilter(mu=np.array([[1., 2.]]).T,
                                  sigma=np.array([[2., 0.],
                                                  [0., 2.]]),
                                  R=0.5,
                                  Q=1.0)

    ground_truths = []
    t = []
    sensor_data = []
    
    x = 2
    alpha_true = 0.1
    T = 20
    
    for i in range(T):
        t.append((i + 1))
        ground_truths.append(np.array([x, alpha_true]))
        sensor_data.append(np.array([np.sqrt(ground_truths[i][0] ** 2 + 1) + np.random.normal(0.0, 1.0)]))
        x = alpha_true*x + np.random.normal(0.0, np.sqrt(0.5))
        
    ground_truths = np.array(ground_truths)
    t = np.array(t)

    kalman.reset()

    predict_means, predict_covariances = kalman.run(np.array(sensor_data))

    plot_prediction(t, ground_truths, predict_means, predict_covariances)


if __name__ == '__main__':
    problem3()
