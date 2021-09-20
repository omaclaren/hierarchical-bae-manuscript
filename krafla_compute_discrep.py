import numpy as np
from lib import GeothermalCore as GC
from lib import InverseCore as IC
import pandas as pd 
import re
import io, pickle

perm_powers_precal = np.log10(np.array([
         1e-15,   1e-15,   1e-15,
         2.5e-16, 1.0e-15, 1.0e-15,
         7.5e-16, 1.5e-15, 1.0e-15,
         1.0e-16, 1.0e-15, 7.5e-16,
         5.0e-14, 1.0e-13, 1.0e-13,
         5.0e-14, 1.0e-13, 1.0e-13, 
         5.0e-14, 1.0e-13, 1.0e-13,
         1.0e-16, 2.5e-16, 1.0e-16, 
         1.0e-13, 2.0e-13, 1.0e-13, 
         1.0e-13, 2.0e-13, 1.0e-13,
         1.0e-13, 2.0e-13, 1.0e-13,
         2.5e-15, 5.0e-16, 2.0e-15, 
         1.0e-15, 2.0e-14, 1.0e-14, 
         2.0e-15, 5.0e-15, 5.0e-15, 
         5.0e-16, 5.0e-16, 5.0e-16,
         5.0e-16, 1.0e-15, 1.0e-15]))

#---coarse process model
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
                     "KJ036", "KJ037",  "KJ039", "KJ040",
                     "KS001", "KV001", "KW001", "KW002", "KJ018" ] #"KJ038",
                     

process_model_coarse.rename_wells_as_obs(list_of_obs_wells)

process_model_coarse.set_rock_permeabilities(perm_powers=perm_powers_precal)

#---fine process model
process_model_fine = GC.GeoModel(name='new_process_model_fine_krafla', 
                                   datfile_name='input-files/medium/krafla_fine',
                                   incon_name='input-files/medium/krafla_fine.incon',
                                   geom_name='input-files/medium/krafla_fine_geom')

process_model_fine.rename_wells_as_obs(list_of_obs_wells)

process_model_fine.set_rock_permeabilities(perm_powers=perm_powers_precal)


# --- data model object
real_data_model = GC.GeoModel(name='new_process_model_krafla', 
                                   datfile_name='input-files/coarse/krafla_vcoarse',
                                   incon_name='input-files/coarse/krafla_vcoarse.incon',
                                   geom_name='input-files/coarse/krafla_vcoarse_geom')

real_data_model.rename_wells_as_obs(list_of_obs_wells)


##---load real data of appropriate resolution and store in above.
real_data_model.d_obs_well = {}
real_data_model.ss_temps_obs_well = {}

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


#param_sets = np.random.normal(loc=perm_powers_precal, scale=0.2, size=(10, 48))
save_path = './saved_data/'
flatchain_filename = 'sampler_flatchain_test_krafla_bayes_model_longerburn_combined.p'


with io.TextIOWrapper(io.open(save_path + flatchain_filename, 'rb')) as f:
    flatchain = pickle.load(f)

param_sets = flatchain

#discrep_model_name = 'test_discrep_kerinci'
discrep_model_name = 'new_discrep_krafla_combined_naive'

discrep = IC.ModelDiscrep(name=discrep_model_name,
    coarse_process_model=process_model_coarse, 
    fine_process_model=process_model_fine, 
    process_space=process_space,
    measurement_space=measurement_space, 
    parameter_set_pool=param_sets)

discrep.compute_raw_model_discrepancy(num_runs=5)

# ---

#model_name = 'test_krafla_bayes_model'
#
#bmodel_coarse = IC.BayesModel(name=model_name,
#                       process_model=process_model_coarse,
#                       fine_process_model=process_model_fine,
#                       measurement_space=measurement_space,
#                       parameter_space=parameter_space,
#                       process_space=process_space)
