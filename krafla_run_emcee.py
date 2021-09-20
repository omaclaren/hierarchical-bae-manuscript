import numpy as np
import timeit
import pickle
import os
import errno
from lib import GeothermalCore as GC
from lib import InverseCore as IC
from scipy.optimize import minimize
import pandas as pd 
from multiprocessing import Pool, cpu_count
import re

perm_powers_precal = np.log10(np.array([
         1e-15,   1e-15,   1e-15,
         2.5e-16, 1.0e-15, 1.0e-15,
         5.0e-16, 1.5e-15, 7.5e-16,
         5.0e-14, 1.5e-13, 5.0e-14,
         5.0e-14, 1.5e-13, 1.5e-13, 
         5.0e-14, 1.5e-13, 1.0e-13,
         1.0e-16, 2.5e-16, 5.0e-17, 
         1.0e-13, 2.25e-13, 1.25e-13, 
         1.0e-13, 2.0e-13, 1.0e-13,
         1.0e-13, 2.0e-13, 1.5e-13,
         1.5e-15, 7.5e-16, 5.0e-15, 
         1.0e-15, 1.5e-14, 2.0e-15, 
         2.0e-15, 5.0e-15, 5.0e-15, 
         7.5e-16, 7.5e-16, 5.0e-16,
         5.0e-16, 1.0e-15, 7.5e-15]))

#---coarse process model
print('Setting up coarse model')
process_model_coarse = GC.GeoModel(name='new_process_model_krafla', 
                                   datfile_name='input-files/coarse/krafla_vcoarse',
                                   incon_name='input-files/coarse/krafla_vcoarse.incon',
                                   geom_name='input-files/coarse/krafla_vcoarse_geom')

list_of_obs_wells = ["IDDP1", "KG003", "KG005", "KG008", "KG010",
                     "KG012", "KG024", "KG025", "KG026", "KJ006", 
                     "KJ007", "KJ009", "KJ011", "KJ013", "KJ014", 
                     "KJ015", "KJ017", "KJ019", "KJ020", "KJ021",
                     "KJ023", "KJ022", "KJ027", "KJ028", "KJ029",
                     "KJ030", "KJ031", "KJ033", "KJ034", "KJ035",
                     "KJ036", "KJ037", "KJ038", "KJ039", "KJ040",
                     "KS001", "KV001", "KW001", "KW002", "KJ018" ]
                     
process_model_coarse.rename_wells_as_obs(list_of_obs_wells)

process_model_coarse.set_rock_permeabilities(perm_powers=perm_powers_precal)

#---fine process model
print('Setting up fine model')
process_model_fine = GC.GeoModel(name='new_process_model_fine_krafla', 
                                   datfile_name='input-files/medium/krafla_fine',
                                   incon_name='input-files/medium/krafla_fine.incon',
                                   geom_name='input-files/medium/krafla_fine_geom')

process_model_fine.rename_wells_as_obs(list_of_obs_wells)

process_model_fine.set_rock_permeabilities(perm_powers=perm_powers_precal)


# --- data model object
print('Setting up data model')
real_data_model = GC.GeoModel(name='new_process_model_krafla', 
                                   datfile_name='input-files/coarse/krafla_vcoarse',
                                   incon_name='input-files/coarse/krafla_vcoarse.incon',
                                   geom_name='input-files/coarse/krafla_vcoarse_geom')

real_data_model.rename_wells_as_obs(list_of_obs_wells)


#---load real data of appropriate resolution and store in above.
real_data_model.d_obs_well = {}
real_data_model.ss_temps_obs_well = {}

print('Loading temperature data')
r = re.compile("OBS*")
obs_wells_list = filter(r.match, process_model_coarse.geom.well.keys())
for i,welli in enumerate(obs_wells_list):
    print(welli)
    df = pd.read_csv('./wells/temperature/elevation/' + welli[4:] + '.csv',header=None,sep=' ', error_bad_lines=False)
    df.rename(columns={0:'d',1:'T'},inplace=True) 

    real_data_model.d_obs_well[i] = df['d']
    real_data_model.ss_temps_obs_well[i] = df['T']

#---create a basic comparison model (basis of likelihood function)
measurement_space = IC.MeasurementSpace(bias=0.0, sigma=10.0)

#---create a parameter model
parameter_space = IC.ParameterSpace(mu=perm_powers_precal, sigma=0.75)


#-----create a basic process space model
process_space = IC.ProcessSpace()

#---create a Bayes model
#use pro_model_coarse for coarse, pro_model_medium for medium
bmodel_coarse = IC.BayesModel(name='test_krafla_bayes_model',
                       process_model=process_model_coarse, 
                       data_model=real_data_model, 
                       measurement_space=measurement_space,
                       parameter_space=parameter_space,
                       process_space=process_space,
                       fine_process_model=process_model_fine)


#---test emcee
#p_start = best_params
p_start = perm_powers_precal
print('Starting MCMC')
sampler = bmodel_coarse.run_emcee(p_start=p_start,
                 n_walkers=100, n_burn=20, n_sample=80, save_data=True, run_name='_' + bmodel_coarse.name)


