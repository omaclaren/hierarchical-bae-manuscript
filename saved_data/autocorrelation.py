# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:52:28 2021

@author: samuels
"""
import numpy as np
from emcee.autocorr import integrated_time
import os
import pickle

chain_filenames = [ 'sampler_chain_test_krafla_bayes_model_discrep.p']

chain = pickle.load(open(chain_filenames[0], "rb"))
tau = np.mean([integrated_time(walker) for walker in chain], axis=0)
