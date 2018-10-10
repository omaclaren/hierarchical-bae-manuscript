import numpy as np
import timeit
import pickle
import os
import errno
from lib import GeothermalCore as GC
from lib import InverseCore as IC
from scipy.optimize import minimize

#---truth
#perm_powers_truths = np.log10(np.array([5.00000000e-14, 1.00000000e-14, 1.00000000e-15,
#                                        5.00000000e-16, 1.00000000e-15, 2.00000000e-14,
#                                        2.50000000e-14, 1.00000000e-14, 5.00000000e-16,
#                                        5.00000000e-16, 5.00000000e-16, 1.00000000e-14]))

# tighter cap
perm_powers_truths = np.log10(np.array([5.00000000e-14, 1.00000000e-14, 1.00000000e-16,
                                        1.00000000e-16, 1.00000000e-15, 2.00000000e-14,
                                        2.50000000e-14, 1.00000000e-14, 5.00000000e-16,
                                        5.00000000e-16, 5.00000000e-16, 1.00000000e-14]))

#---coarse process model
process_model_coarse = GC.GeoModel(name='test_process_model_coarse',
                                   datfile_name='input-files/elvar-new/coarse-model/2DC002',
                                   incon_name='input-files/elvar-new/coarse-model/2DC002_IC',
                                   geom_name='input-files/elvar-new/coarse-model/g2coarse')

#---medium process model
process_model_medium = GC.GeoModel(name='test_process_model_medium',
                                   datfile_name='input-files/elvar-new-tighter-cap/medium-model/2DM002',
                                   incon_name='input-files/elvar-new-tighter-cap/medium-model/2DM002_IC',
                                   geom_name='input-files/elvar-new-tighter-cap/medium-model/g2medium')

#---fine synthetic model
process_model_fine = GC.GeoModel(name='test_process_model_fine',
                                 datfile_name='input-files/elvar-new-tighter-cap/fine-model/2DF002',
                                 incon_name='input-files/elvar-new-tighter-cap/fine-model/2DF002_IC',
                                 geom_name='input-files/elvar-new-tighter-cap/fine-model/g2fine')

comparison_model = IC.ComparisonModel(bias=0.0, sigma=5.0)

#sflat = pickle.load(
#    open("./saved_data/sampler_flatchain_test_process_model_coarse.p", "rb"))

sflat = pickle.load(
    open("./saved_data/sampler_flatchain_test_process_model_medium.p", "rb"))

param_sets = sflat
#param_sets = np.random.normal(loc=-15, scale=1.5, size=(30, 12))
#param_sets[:, 2:4] = -16

discrep = IC.ModelDiscrep(
    coarse_process_model=process_model_medium, fine_process_model=process_model_fine, 
    comparison_model=comparison_model, parameter_set_pool=param_sets)

discrep.compute_raw_model_discrepancy(num_runs=200)

# ---
parameter_model = IC.ParameterModel()
bmodel = IC.BayesModel(name='test_bayes_model_discrep_med_fine',
                       process_model=process_model_medium,
                       data_model=process_model_fine,
                       comparison_model=comparison_model,
                       parameter_model=parameter_model)
