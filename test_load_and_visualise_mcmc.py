import numpy as np
import timeit
import pickle
import os
import errno
from lib import GeothermalCore as GC
from lib import InverseCore as IC

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
synthetic_model_fine = GC.GeoModel(name='test_synthetic_model_fine',
                                   datfile_name='input-files/elvar-new-tighter-cap/fine-model/2DF002',
                                   incon_name='input-files/elvar-new-tighter-cap/fine-model/2DF002_IC',
                                   geom_name='input-files/elvar-new-tighter-cap/fine-model/g2fine')

#---generate or load synthetic data
generate_new_data = False
if generate_new_data:
    synthetic_model_fine.generate_synthetic_data(
        perm_powers_truths=perm_powers_truths)
else:
    synthetic_data = pickle.load(open("./saved_data/synthetic_data.p", "rb"))
    synthetic_model_fine.ss_temps = synthetic_data['T_measured']
    #syn_model.ss_temps = syn_model.T_measured
    synthetic_model_fine.T_noise = synthetic_data['T_noise']
    synthetic_model_fine.ss_temps_obs_well = synthetic_data['T_obs_well']
    synthetic_model_fine.d_obs_well = synthetic_data['d_obs_well']

#---create a parameter space model
parameter_model = IC.ParameterModel(mu=-15, sigma=1.5)

#---create or load comparison space model (basis of likelihood function)
load_discrepancy = True
#discrepancy_filename = 'discrepancies_ru.p'
discrepancy_filename = 'discrepancies_combined.p'

#pre-discrep.
comparison_model = IC.ComparisonModel(bias=0.0, sigma=5.0) #without discrep.

if load_discrepancy:
    discrep = IC.ModelDiscrep(comparison_model=comparison_model)
    discrep_data = pickle.load(open("./saved_data/" + discrepancy_filename, "rb"))
    discrep.discrepancy_dist = discrep_data
    discrep.compute_normal_discrep_model()
    #update comparison model based on discrep.
    comparison_model.bias = discrep.discrepancy_mean
    comparison_model.icov = discrep.combined_icov   

#---create a Bayes model
#use process_model_coarse for coarse, process_model_medium for medium.
bmodel = IC.BayesModel(name='test_bayes_model_med_fine_tighter_cap_visualise',
                       process_model=process_model_medium,
                       data_model=synthetic_model_fine,
                       comparison_model=comparison_model,
                       parameter_model=parameter_model)

#---load mcmc chain and choose subset for predictive plots
#flatchain_filename = 'sampler_flatchain_test_bayes_model_discrep_med_fine_ru_16jan.p'
flatchain_filename = 'sampler_flatchain_test_bayes_model_med_fine_tighter_cap_mcmc_with_combined_discrep_incl_outliers.p'

flatchain = pickle.load(open("./saved_data/" + flatchain_filename, "rb"))
param_sets = parameter_model.choose_random_parameter_subsets(parameter_pool=flatchain, n_subsets=50)

bmodel.sampler_flatchain = flatchain

#bmodel.sampler_flatchain = flatchain

#---predictive checks 
# with bias correction, no re-map
bmodel.predictive_checks(parameter_sets=param_sets, subtract_bias=True)
# without bias correction, no re-map
bmodel.predictive_checks(parameter_sets=param_sets, subtract_bias=False)

# with bias correction, with re-map
bmodel.predictive_checks(parameter_sets=param_sets, subtract_bias=True, do_map_fine_to_coarse=True)
# without bias correction, with re-map
bmodel.predictive_checks(parameter_sets=param_sets, subtract_bias=False, do_map_fine_to_coarse=True)
