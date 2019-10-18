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
                                   geom_name = 'input-files/elvar-new/coarse-model/g2coarse',
                                   islayered=True)

#---medium process model
process_model_medium = GC.GeoModel(name='test_process_model_medium',
                                   datfile_name='input-files/elvar-new-tighter-cap/medium-model/2DM002',
                                   incon_name='input-files/elvar-new-tighter-cap/medium-model/2DM002_IC',
                                   geom_name='input-files/elvar-new-tighter-cap/medium-model/g2medium',
                                   islayered=True)

#---fine process model
process_model_fine = GC.GeoModel(name='test_process_model_fine',
                                   datfile_name='input-files/elvar-new-tighter-cap/fine-model/2DF002',
                                   incon_name='input-files/elvar-new-tighter-cap/fine-model/2DF002_IC',
                                   geom_name='input-files/elvar-new-tighter-cap/fine-model/g2fine',
                                   islayered=True)

#---fine synthetic model
synthetic_model_fine = GC.GeoModel(name='test_synthetic_model_fine', 
                        datfile_name='input-files/elvar-new-tighter-cap/fine-model/2DF002', 
                        incon_name='input-files/elvar-new-tighter-cap/fine-model/2DF002_IC', 
                        geom_name = 'input-files/elvar-new-tighter-cap/fine-model/g2fine',
                        islayered=True)




#---create a basic comparison model (basis of likelihood function)
measurement_space = IC.MeasurementSpace(bias=0.0, sigma=5.0)

#just default/null process and parameter spaces required.
process_space = IC.ProcessSpace()
parameter_space = IC.ParameterSpace()

#sflat = pickle.load(
#    open("./saved_data/sampler_flatchain_test_process_model_coarse.p", "rb"))

sflat = pickle.load(
    open("./saved_data/sampler_flatchain_test_process_model_medium.p", "rb"))

param_sets = sflat
#param_sets = np.random.normal(loc=-15, scale=1.5, size=(30, 12))
#param_sets[:, 2:4] = -16

discrep = IC.ModelDiscrep(name='test_discrep_med_fine_refactor',
    coarse_process_model=process_model_medium, fine_process_model=process_model_fine, 
    process_space=process_space,
    measurement_space=measurement_space, parameter_set_pool=param_sets)

discrep.compute_raw_model_discrepancy(num_runs=10,save_data=False)

# ---
bmodel = IC.BayesModel(name='test_bayes_model_discrep_med_fine_refactor',
                       process_model=process_model_medium,
                       fine_process_model=process_model_fine,
                       measurement_space=measurement_space,
                       parameter_space=parameter_space,
                       process_space=process_space)
