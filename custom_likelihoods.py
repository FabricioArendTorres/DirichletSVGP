#!/usr/bin/env
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import gpflow

tfd = tfp.distributions
gpflow.models.SGPMC

class Dirichlet(gpflow.likelihoods.MonteCarloLikelihood):
    """
    Make it an instance of MonteCarloLikelihood, since the dimension of the
    variational expectation is equal to the number of classes
    --> prefer MCMC instead of quadrature over integration of high dimensional spaces
    """

    def __init__(self, invlink=tf.exp, **kwargs):
        """
        Initalize Dirichlet Distribution.
        I use a log/exp link, since this fits nicely with a zero-mean GP prior,
         resulting in a Dir(exp(0)) = Dir(1), i.e. uniform probability within the simplex.
        :param invlink:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.invlink = invlink
        self.num_monte_carlo_points = 100

    @staticmethod
    def _log_prob(concentration, y):
        """
        Dirichlet log pdf.
        This is essentially just the tensorflow probability implementation, minus the whole overhead
        :param concentration:
        :param y:
        :return:
        """
        return (tf.reduce_sum(tf.math.xlogy(concentration - 1., y), axis=-1) -
                tf.math.lbeta(concentration))

    @staticmethod
    def _sample_dir(concentration, n, seed=None):
        """
        Sample from dirichlet distribution.
        Again, the tensorflow probability implementation minus overhead.
        :param concentration:
        :param n:
        :param seed:
        :return:
        """
        gamma_sample = tf.random.gamma(
            shape=[n], alpha=concentration, dtype=gpflow.default_float(), seed=seed)
        return gamma_sample / tf.reduce_sum(gamma_sample, axis=-1, keepdims=True)

    def log_prob(self, F, Y):
        """
        return log_pdf of dirichlet. Required for calculating the Integrals of the
        variational expectation (in our case via MCMC) in GPFlow.
        AFAIK no closed form of E_q(f)[log p(y|f)] for a dirichlet likelihood exists,
        or at least it should not be that easy to derive..
        :param F:
        :param Y:
        :return:
        """
        return self._log_prob(concentration=self.invlink(F), y=Y)

    def conditional_mean(self, F):
        """
        Expectation of the Dirichlet, given the latent value (which parameterize the concentrations in our case).
        https://en.wikipedia.org/wiki/Dirichlet_distribution#Moments
        :param F:
        :return:
        """
        concentration = self.invlink(F)
        concentration_sum = tf.reduce_sum(concentration, axis=-1)[..., tf.newaxis]
        return concentration / concentration_sum

    def conditional_variance(self, F):
        """
        Variance of the Dirichlet, given the latent value (which parameterizes the concentration in our case).
        https://en.wikipedia.org/wiki/Dirichlet_distribution#Moments
        :param F:
        :return:
        """
        concentration = self.invlink(F)
        c_sum = tf.reduce_sum(concentration, axis=-1)[..., tf.newaxis]
        return concentration * (c_sum - concentration) / (c_sum ** 2 * (c_sum + 1))
