# Copyright 2023 Jiantong Jiang, The University of Western Australia
# Licensed under the Apache License, Version 2.0 (the "License").
#
# This file is created by Jiantong Jiang to implement FastBO
# Design new searcher FastBOSearcher and new methods for the searcher.
# Add a dataclass ``
#
# --- Original Apache License and Copyright ---
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import List

import numpy as np
from scipy.stats import norm


def neg_log_likelihood(params, x, y):
    """
    Use Maximum Likelihood Estimation (MLE) for parameter estimation.
    Return the negative log-likelihood function to be minimized, given
    the observations of x and y.
    The parametric learning curve model is the combination of three
    parametric learning curve models: pow3, exp3, log2.
    """
    w1, c1, a1, alpha1, w3, c3, a3, b3, w5, c5, a5, sigma = params
    mean = w1 * (c1 - a1 * np.array(x) ** (-alpha1)) + \
           w3 * (c3 - np.exp(-a3 * np.array(x) + b3)) + \
           w5 * (c5 - a5 * np.log(np.array(x)))
    log_likelihood = -np.sum(norm.logpdf(y, loc=mean, scale=sigma))
    return log_likelihood


def neg_log_likelihood_pow3(params, x, y):
    """
    Use Maximum Likelihood Estimation (MLE) for parameter estimation.
    Return the negative log-likelihood function to be minimized, given
    the observations of x and y.
    Use pow3 only if the combined parametric learning model failed to
    pass the simple validation test.
    """
    c1, a1, alpha1, sigma = params
    mean = c1 - a1 * np.array(x) ** (-alpha1)
    log_likelihood = -np.sum(norm.logpdf(y, loc=mean, scale=sigma))
    return log_likelihood


def combined_function(params, x):
    return params[0] * (params[1] - params[2] * x ** (-params[3])) + \
           params[4] * (params[5] - np.exp(-params[6] * x + params[7])) + \
           params[8] * (params[9] - params[10] * np.log(x))


def pow3(params, x):
    return params[0] - params[1] * x ** (-params[2])


def log2(params, x):
    return params[8] * (params[9] - params[10] * np.log(x))


def neg_log_likelihood_considered(params, x, y):
    """
    This function includes all 8 parametric learning curve models we
    considered: pow3, pow4, exp3, exp4, log2, ilog2, weibull, mmf.
    """
    w1, c1, a1, alpha1, \
        w2, c2, a2, b2, alpha2, \
        w3, c3, a3, b3, \
        w4, c4, a4, b4, alpha4, \
        w5, c5, a5, \
        w6, c6, a6, \
        w7, c7, b7, a7, alpha7, \
        w8, a8, b8, c8, d8, \
        sigma = params
    mean = w1 * (c1 - a1 * np.array(x) ** (-alpha1)) + \
           w2 * (c2 - (a2 * np.array(x) + b2) ** (-alpha2)) + \
           w3 * (c3 - np.exp(-a3 * np.array(x) + b3)) + \
           w4 * (c4 - np.exp(-a4 * (np.array(x) ** alpha4) + b4)) + \
           w5 * (c5 - a5 * np.log(np.array(x))) + \
           w6 * (c6 - a6 / np.log(np.array(x + 1))) + \
           w7 * (c7 - b7 * np.exp(-a7 * (np.array(x) ** alpha7))) + \
           w8 * ((a8 * b8 + c8 * np.array(x) ** d8) / (b8 + np.array(x) ** d8))
    log_likelihood = -np.sum(norm.logpdf(y, loc=mean, scale=sigma))
    return log_likelihood


def neg_log_likelihood_min(params, x, y):
    """
    Use Maximum Likelihood Estimation (MLE) for parameter estimation.
    Return the negative log-likelihood function to be minimized, given
    the observations of x and y.
    The parametric learning curve model is the combination of three
    parametric learning curve models: pow3, exp3, log2.
    """
    w1, c1, a1, alpha1, w3, c3, a3, b3, w5, c5, a5, sigma = params
    mean = w1 * (c1 + a1 * np.array(x) ** (-alpha1)) + \
           w3 * (c3 + np.exp(-a3 * np.array(x) + b3)) + \
           w5 * (c5 + a5 * np.log(np.array(x)))
    log_likelihood = -np.sum(norm.logpdf(y, loc=mean, scale=sigma))
    return log_likelihood


def neg_log_likelihood_pow3_min(params, x, y):
    """
    Use Maximum Likelihood Estimation (MLE) for parameter estimation.
    Return the negative log-likelihood function to be minimized, given
    the observations of x and y.
    Use pow3 only if the combined parametric learning model failed to
    pass the simple validation test.
    """
    c1, a1, alpha1, sigma = params
    mean = c1 + a1 * np.array(x) ** (-alpha1)
    log_likelihood = -np.sum(norm.logpdf(y, loc=mean, scale=sigma))
    return log_likelihood


def combined_function_min(params, x):
    return params[0] * (params[1] + params[2] * x ** (-params[3])) + \
           params[4] * (params[5] + np.exp(-params[6] * x + params[7])) + \
           params[8] * (params[9] + params[10] * np.log(x))


def pow3_min(params, x):
    return params[0] + params[1] * x ** (-params[2])


def log2_min(params, x):
    return params[8] * (params[9] + params[10] * np.log(x))
