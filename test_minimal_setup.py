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
                                   geom_name = 'input-files/elvar-new/coarse-model/g2coarse')

#---medium process model
process_model_medium = GC.GeoModel(name='test_process_model_medium',
                                   datfile_name='input-files/elvar-new-tighter-cap/medium-model/2DM002',
                                   incon_name='input-files/elvar-new-tighter-cap/medium-model/2DM002_IC',
                                   geom_name='input-files/elvar-new-tighter-cap/medium-model/g2medium')

#---fine process model
process_model_fine = GC.GeoModel(name='test_process_model_fine',
                                   datfile_name='input-files/elvar-new-tighter-cap/fine-model/2DF002',
                                   incon_name='input-files/elvar-new-tighter-cap/fine-model/2DF002_IC',
                                   geom_name='input-files/elvar-new-tighter-cap/fine-model/g2fine')

#---fine synthetic model
synthetic_model_fine = GC.GeoModel(name='test_synthetic_model_fine', 
                        datfile_name='input-files/elvar-new-tighter-cap/fine-model/2DF002', 
                        incon_name='input-files/elvar-new-tighter-cap/fine-model/2DF002_IC', 
                        geom_name = 'input-files/elvar-new-tighter-cap/fine-model/g2fine')

#---generate or load synthetic data
generate_new_data = False
if generate_new_data:
    synthetic_model_fine.generate_synthetic_data(perm_powers_truths=perm_powers_truths)
else:
    synthetic_data = pickle.load(open("./saved_data/synthetic_data.p", "rb"))
    synthetic_model_fine.ss_temps = synthetic_data['T_measured']
    #syn_model.ss_temps = syn_model.T_measured
    synthetic_model_fine.T_noise = synthetic_data['T_noise']
    synthetic_model_fine.ss_temps_obs_well = synthetic_data['T_obs_well']
    synthetic_model_fine.d_obs_well = synthetic_data['d_obs_well']

#---create a basic comparison model (basis of likelihood function)
comparison_model = IC.ComparisonModel(bias=0.0, sigma=5.0)

#---create a parameter model
parameter_model = IC.ParameterModel(mu=-15, sigma=1.5)

#---create a Bayes model
#use pro_model_coarse for coarse, pro_model_medium for medium
#bmodel = IC.BayesModel(name='test_bayes_model',
#                       process_model=process_model_medium, 
#                       data_model=synthetic_model_fine, 
#                       comparison_model=comparison_model,
#                       parameter_model=parameter_model)

bmodel = IC.BayesModel(name='test_bayes_model',
                       process_model=process_model_fine, 
                       data_model=synthetic_model_fine, 
                       comparison_model=comparison_model,
                       parameter_model=parameter_model)

#---test calculate likelihood
start = timeit.default_timer()

bmodel.lnlike(perm_powers_current=perm_powers_truths)

stop = timeit.default_timer()
print('time to compute lnlike (s): ')
print(stop - start)

#---test calculate model run time
start = timeit.default_timer()

bmodel.process_model.simulate()

stop = timeit.default_timer()
print('time to run model (s): ')
print(stop - start)


#sflat = pickle.load(
#    open("./saved_data/sampler_flatchain_test_process_model_coarse.p", "rb"))

#discrep = IC.ModelDiscrep(
#    coarse_process_model=process_model_coarse, fine_process_model=process_model_medium, 
#    comparison_model=comparison_model, parameter_set_pool=sflat)
