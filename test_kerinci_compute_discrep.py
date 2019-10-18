import numpy as np
import timeit
import pickle
import os
import errno
from lib import GeothermalCore as GC
from lib import InverseCore as IC
from scipy.optimize import minimize
import pandas as pd 

perm_powers_precal = np.log10(np.array([
        1.00000000e-15,   1.00000000e-15,   1.00000000e-15,
        2.00000000e-15,   2.00000000e-15,   8.00000000e-15,
        5.00000000e-16,   5.00000000e-16,   5.00000000e-16,
        1.00000000e-16,   1.00000000e-16,   1.00000000e-16,
        5.00000000e-14,   5.00000000e-14,   5.00000000e-16,
        5.00000000e-14,   5.00000000e-14,   5.00000000e-16,
        5.00000000e-14,   1.00000000e-14,   3.00000000e-14,
        5.00000000e-15,   5.00000000e-15,   5.00000000e-15,
        5.00000000e-14,   5.00000000e-14,   5.00000000e-16,
        5.00000000e-14,   5.00000000e-14,   5.00000000e-16]))

save_path = './saved_data/'

#---coarse process model
process_model_coarse = GC.GeoModel(name='test_process_model_kerinci', 
                                   datfile_name='input-files/kerinci/coarse-model/Keriv0_027',
                                   incon_name='input-files/kerinci/coarse-model/Keriv0_027',
                                   geom_name='input-files/kerinci/coarse-model/gKerinci_v0')

list_of_obs_wells = ['LP002','LP001','KRDB1']

process_model_coarse.rename_wells_as_obs(list_of_obs_wells)

process_model_coarse.set_rock_permeabilities(perm_powers=perm_powers_precal)

#list_of_obs_wells = ['LP002'] #['LP002','LP001','KRDB1']'KRDB1']

#---fine process model
process_model_fine = GC.GeoModel(name='test_process_model_kerinci_fine', 
                                   datfile_name='input-files/kerinci/fine-model/Keriv1_027',
                                   incon_name='input-files/kerinci/fine-model/Keriv1_027',
                                   geom_name='input-files/kerinci/fine-model/gKerinci_v1')

process_model_fine.rename_wells_as_obs(list_of_obs_wells)

process_model_fine.set_rock_permeabilities(perm_powers=perm_powers_precal)

# ---Layered
# perm_powers_truths = np.log10(np.array([
#         1.00000000e-15,   1.00000000e-15,
#         2.00000000e-15,   8.00000000e-15,
#         5.00000000e-16,   5.00000000e-16,
#         1.00000000e-16,   1.00000000e-16,
#         5.00000000e-14,   5.00000000e-16,
#         5.00000000e-14,   5.00000000e-16,
#         5.00000000e-14,   3.00000000e-14,
#         5.00000000e-15,   5.00000000e-15,
#         5.00000000e-14,   5.00000000e-16,
#         5.00000000e-14,   5.00000000e-16]))
#
# process_model_coarse = GC.GeoModel(name='test_process_model_kericini', 
#                                    datfile_name='input-files/kericini/coarse-model/Keriv0_027',
#                                    incon_name='input-files/kericini/coarse-model/Keriv0_027',
#                                    geom_name='input-files/kericini/coarse-model/gKerinci_v0',
#                                    islayered=True)


# --- data model object
real_data_model = GC.GeoModel(name='test_data_model_kerinci', 
                                   datfile_name='input-files/kerinci/coarse-model/Keriv0_027',
                                   incon_name='input-files/kerinci/coarse-model/Keriv0_027',
                                   geom_name='input-files/kerinci/coarse-model/gKerinci_v0')

real_data_model.rename_wells_as_obs(list_of_obs_wells)

#---load real data of appropriate resolution and store in above.

real_data_model.d_obs_well = {}
real_data_model.ss_temps_obs_well = {}

for i,welli in enumerate(list_of_obs_wells):

    df = pd.read_csv(save_path + 'kerinci_data/Temp_' + welli + '.dat',header=None,sep=' ')
    df.rename(columns={1:'d',0:'T'},inplace=True) 

    real_data_model.d_obs_well[i] = df['d']
    real_data_model.ss_temps_obs_well[i] = df['T']

#---create a basic comparison model (basis of likelihood function)
measurement_space = IC.MeasurementSpace(bias=0.0, sigma=10.0)

#---create a parameter model
parameter_space = IC.ParameterSpace(mu=-15, sigma=1.5)


#-----create a basic process space model
process_space = IC.ProcessSpace()

#---create a Bayes model
#use pro_model_coarse for coarse, pro_model_medium for medium
#bmodel = IC.BayesModel(name='test_bayes_model',
#                       process_model=process_model_medium, 
#                       data_model=synthetic_model_fine, 
#                       comparison_model=comparison_model,
#                       parameter_model=parameter_model)

#flatchain_filename = 'sampler_flatchain_test_kerinci_bayes_model.p'
flatchain_filename = 'sampler_flatchain_test_kerinci_bayes_model_combined.p'

flatchain = pickle.load(open(save_path + flatchain_filename, "rb"))
#param_sets = parameter_space.choose_random_parameter_subsets(parameter_pool=flatchain, n_subsets=50)

#bmodel_coarse.sampler_flatchain = flatchain

param_sets = flatchain
#param_sets = np.random.normal(loc=-15, scale=1.5, size=(30, 12))
#param_sets[:, 2:4] = -16

#discrep_model_name = 'test_discrep_kerinci'
discrep_model_name = 'test_discrep_kerinci_combined_naive'

discrep = IC.ModelDiscrep(name=discrep_model_name,
    coarse_process_model=process_model_coarse, 
    fine_process_model=process_model_fine, 
    process_space=process_space,
    measurement_space=measurement_space, 
    parameter_set_pool=param_sets)

discrep.compute_raw_model_discrepancy(num_runs=100)

# ---

model_name = 'test_kerinci_bayes_model'

bmodel_coarse = IC.BayesModel(name=model_name,
                       process_model=process_model_coarse, 
                       data_model=real_data_model, 
                       measurement_space=measurement_space,
                       parameter_space=parameter_space,
                       process_space=process_space,
                       fine_process_model=process_model_fine)
