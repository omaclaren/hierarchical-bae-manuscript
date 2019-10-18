import numpy as np
import timeit
import pickle
import os
import errno
from lib import GeothermalCore as GC
from lib import InverseCore as IC
from scipy.optimize import minimize
import re
import pandas as pd

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
#perm_powers_truths = np.log10(np.array([5.00000000e-14, 1.00000000e-14, 1.00000000e-16,
#                                        1.00000000e-16, 1.00000000e-15, 2.00000000e-14,
#                                        5.00000000e-16, 5.00000000e-16, 1.00000000e-14]))
#                                        2.50000000e-14, 1.00000000e-14, 5.00000000e-16,

perm_powers_truths = np.log10(np.array([
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

#---coarse process model
process_model_coarse = GC.GeoModel(name='test_process_model_kerinci', 
                                   datfile_name='input-files/kerinci/coarse-model/Keriv0_027',
                                   incon_name='input-files/kerinci/coarse-model/Keriv0_027',
                                   geom_name='input-files/kerinci/coarse-model/gKerinci_v0')

list_of_obs_wells = ['LP002','LP001','KRDB1']

process_model_coarse.rename_wells_as_obs(list_of_obs_wells)

process_model_coarse.set_rock_permeabilities(perm_powers=perm_powers_truths)

#list_of_obs_wells = ['LP002'] #['LP002','LP001','KRDB1']'KRDB1']

#---fine process model
process_model_fine = GC.GeoModel(name='test_process_model_kerinci_fine', 
                                   datfile_name='input-files/kerinci/fine-model/Keriv1_027',
                                   incon_name='input-files/kerinci/fine-model/Keriv1_027',
                                   geom_name='input-files/kerinci/fine-model/gKerinci_v1')

process_model_fine.rename_wells_as_obs(list_of_obs_wells)

process_model_fine.set_rock_permeabilities(perm_powers=perm_powers_truths)

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

    df = pd.read_csv('./saved_data/kerinci_data/Temp_' + welli + '.dat',header=None,sep=' ')
    df.rename(columns={1:'d',0:'T'},inplace=True) 

    real_data_model.d_obs_well[i] = df['d']
    real_data_model.ss_temps_obs_well[i] = df['T']

#---create a basic comparison model (basis of likelihood function)
measurement_space = IC.MeasurementSpace(bias=0.0, sigma=5.0)

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

bmodel_coarse = IC.BayesModel(name='test_bayes_model',
                       process_model=process_model_coarse, 
                       data_model=real_data_model, 
                       measurement_space=measurement_space,
                       parameter_space=parameter_space,
                       process_space=process_space,
                       fine_process_model=process_model_fine)

#---test calculate likelihood
start = timeit.default_timer()

ll = bmodel_coarse.lnlike(perm_powers_current=perm_powers_truths)

stop = timeit.default_timer()
print('time to compute lnlike (s): ')
print(stop - start)

print('ll:')
print(ll)

#---test calculate model run time
start = timeit.default_timer()

bmodel_coarse.process_model.simulate()

stop = timeit.default_timer()
print('time to run model (s): ')
print(stop - start)

#---create a Bayes model with coarse data fine model
#use pro_model_coarse for coarse, pro_model_medium for medium
#bmodel = IC.BayesModel(name='test_bayes_model',
#                       process_model=process_model_medium, 
#                       data_model=synthetic_model_fine, 
#                       comparison_model=comparison_model,
#                       parameter_model=parameter_model)

#bmodel_coarse.predictive_checks(parameter_sets=10*[perm_powers_truths],do_data_space=True,subtract_bias=False)
#bmodel_coarse.predictive_checks(parameter_sets=10*[perm_powers_truths],do_data_space=False,subtract_bias=False)

#bmodel_coarse.predictive_checks(parameter_sets=10*[perm_powers_truths],do_data_space=True,subtract_bias=True)
#bmodel_coarse.predictive_checks(parameter_sets=10*[perm_powers_truths],do_data_space=False,subtract_bias=True)

#bmodel_coarse.predictive_checks(parameter_sets=10*[perm_powers_truths],do_data_space=False,subtract_bias=False)


#---

bmodel_fine = IC.BayesModel(name='test_bayes_model',
                       process_model=process_model_fine, 
                       data_model=real_data_model, 
                       measurement_space=measurement_space,
                       parameter_space=parameter_space,
                       process_space=process_space,
                       fine_process_model=process_model_fine)

#---test calculate likelihood
start = timeit.default_timer()

ll = bmodel_fine.lnlike(perm_powers_current=perm_powers_truths)

stop = timeit.default_timer()
print('time to compute lnlike fine (s): ')
print(stop - start)

print('ll:')
print(ll)

#---test calculate model run time
start = timeit.default_timer()

bmodel_fine.process_model.simulate()

stop = timeit.default_timer()
print('time to run model fine (s): ')
print(stop - start)

#---create a Bayes model with coarse data fine model
#use pro_model_coarse for coarse, pro_model_medium for medium
#bmodel = IC.BayesModel(name='test_bayes_model',
#                       process_model=process_model_medium, 
#                       data_model=synthetic_model_fine, 
#                       comparison_model=comparison_model,
#                       parameter_model=parameter_model)

#bmodel_fine.predictive_checks(parameter_sets=[perm_powers_truths],do_data_space=True,subtract_bias=False)
#bmodel_fine.predictive_checks(parameter_sets=[perm_powers_truths],do_data_space=True,subtract_bias=True)