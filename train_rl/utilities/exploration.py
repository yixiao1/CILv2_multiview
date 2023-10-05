import numpy as np

from math import sqrt

class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma, theta=0.15, dt=0.1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_sigma, desired_action_sigma, adaptation_coefficient):
        """
        Note that initial_sigma and current_sigma refer to std of parameter noise,
        but desired_action_sigma refers to (as name notes) desired std in action space
        """
        self.initial_sigma = initial_sigma
        self.desired_action_sigma = desired_action_sigma
        self.adaptation_coefficient = adaptation_coefficient

        self.current_sigma = initial_sigma

    def adapt(self, distance):
        if distance > self.desired_action_sigma:
            # Decrease sigma.
            self.current_sigma /= self.adaptation_coefficient
        else:
            # Increase sigma.
            self.current_sigma *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_sigma': self.current_sigma,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_sigma={}, desired_action_sigma={}, adaptation_coefficient={})'
        return fmt.format(self.initial_sigma, self.desired_action_sigma, self.adaptation_coefficient)

def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist