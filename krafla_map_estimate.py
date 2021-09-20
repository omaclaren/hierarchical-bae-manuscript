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
         1e-15,   1e-15,   1e-15,
         1.4e-16, 9.8e-16, 1.5e-16,
         7.5e-16, 1.5e-15, 1.0e-15,
         5.0e-16, 2.0e-15, 1.5e-15,
         4.0e-13, 4.0e-13, 4.0e-13,
         4.0e-14, 3.0e-13, 4.0e-13, 
         1.0e-13, 1.0e-13, 1.0e-13,
         1.0e-16, 5.0e-16, 1.0e-16, 
         8.0e-14, 2.0e-13, 2.0e-13, 
         8.0e-14, 2.0e-13, 2.0e-13,
         6.0e-14, 1.0e-13, 1.0e-13,
         4.7e-14, 6.5e-16, 1.0e-14, 
         1.2e-15, 9.7e-16, 1.7e-16, 
         5.0e-16, 7.5e-16, 7.0e-15, 
         1.0e-16, 2.5e-16, 2.5e-16,
         1.0e-16, 2.3e-16, 2.9e-15]))

#perm_powers_precal = {
#               'BASE1': np.log10(np.array([7.5e-16, 1.5e-15, 1.0e-15])),
#               'BARFT': np.log10(np.array([1.4e-16, 9.8e-16, 1.5e-16])),
#               'HVERA': np.log10(np.array([5.0e-16, 2.0e-15, 1.5e-15])),
#               'HYOO1': np.log10(np.array([4.4e-14, 1.0e-13, 4.0e-13])),
#               'HYOO2': np.log10(np.array([6.0e-14, 1.0e-13, 4.0e-13])),
#               'HYOO3': np.log10(np.array([8.0e-14, 3.0e-13, 4.0e-13])),
#               'HYOO4': np.log10(np.array([1.0e-13, 5.0e-13, 5.0e-13])),
#               'LVOO1': np.log10(np.array([1.0e-16, 5.0e-16, 1.0e-16])),
#               'LVOO2': np.log10(np.array([4.0e-14, 1.0e-13, 1.0e-13])),
#               'LVOO3': np.log10(np.array([4.0e-14, 1.0e-13, 6.0e-14])),
#               'LVOO4': np.log10(np.array([6.0e-14, 1.0e-13, 7.0e-13])),
#               'TVFLT': np.log10(np.array([4.7e-14, 6.5e-16, 1.0e-14])),
#               'OUTER': np.log10(np.array([1.2e-15, 1.0e-15, 1.0e-16])),
#               'CLAYC': np.log10(np.array([1.0e-15, 1.5e-15, 8.0e-15])),
#               'BASE2': np.log10(np.array([1.0e-16, 2.5e-16, 2.5e-16])),
#               'OUTBS': np.log10(np.array([1.0e-16, 2.3e-16, 2.9e-15])),}


save_path = './saved_data/'

#---coarse process model
process_model_coarse = GC.GeoModel(name='test_process_model_krafla', 
                                   datfile_name='input-files/sc/coarse/newold26',
                                   incon_name='input-files/sc/coarse/save27.incon',
                                   geom_name='input-files/sc/coarse/gkrafla_v04_fewwells')

#list_of_obs_wells = ['KG008', 'KG005', 'KJ011', 'KG026', 'KJ021']
list_of_obs_wells = ["IDDP1", "KG003", "KG005", "KG008", "KG010",
                     "KG024", "KG026", "KJ006", "KJ007", "KJ009", 
                     "KJ011", "KJ013", "KJ014", "KJ015", "KJ017",
                     "KJ019", "KJ020", "KJ021", "KJ027", "KJ028",
                     "KJ029", "KJ030", "KJ031", "KJ033", "KJ034", 
                     "KJ035", "KJ036", "KJ037", "KJ039", "KJ040",
                     "KW001", "KW002"]


process_model_coarse.rename_wells_as_obs(list_of_obs_wells)

process_model_coarse.set_rock_permeabilities(perm_powers=perm_powers_precal)

        #rt.permeability = np.power(10,perm_powers[self.rocktype_to_perm_power_index[str(rt)]])

#---fine process model
#process_model_fine = GC.GeoModel(name='new_process_model_krafla_fine', 
#                                   datfile_name='input-files/medium/krafla_v04_fine',
#                                   incon_name='input-files/medium/krafla_v04_fine.incon',
#                                   geom_name='input-files/medium/gkrafla_v04_fine')


#process_model_fine.rename_wells_as_obs(list_of_obs_wells)

#process_model_fine.set_rock_permeabilities(perm_powers=perm_powers_precal)


# --- data model object
real_data_model = GC.GeoModel(name='test_process_model_krafla', 
                                   datfile_name='input-files/sc/coarse/newold26',
                                   incon_name='input-files/sc/coarse/save27.incon',
                                   geom_name='input-files/sc/coarse/gkrafla_v04_fewwells')


real_data_model.rename_wells_as_obs(list_of_obs_wells)


#---load real data of appropriate resolution and store in above.
real_data_model.d_obs_well = {}
real_data_model.ss_temps_obs_well = {}

r = re.compile("OBS*")
obs_wells_list = filter(r.match, process_model_coarse.geom.well.keys())
for i,welli in enumerate(obs_wells_list):
    print(welli)
    df = pd.read_csv('./wells/temperature/elevation/new/' + welli[4:] + '.csv',header=None,sep=' ', error_bad_lines=False)
    df.rename(columns={0:'d',1:'T'},inplace=True) 

    real_data_model.d_obs_well[i] = df['d']
    real_data_model.ss_temps_obs_well[i] = df['T']

#---create a basic comparison model (basis of likelihood function)
measurement_space = IC.MeasurementSpace(bias=0.0, sigma=10.0)

#---create a parameter model
parameter_space = IC.ParameterSpace(mu=perm_powers_precal, sigma=0.5)


#-----create a basic process space model
process_space = IC.ProcessSpace()

#-----set up discrepancy info
load_discrepancy = False
map_coarse_discrep_to_data_grid = False  # - default should be False?
discrepancy_filename = 'discrepancies_new_discrep_krafla_combined_naive.p'# 'discrepancies_combined_kerinci.p'
#discrepancy_filename = 'discrepancies_combined_krafla_map_data.p'
discrep_model_name = 'test_discrep_krafla'

if load_discrepancy:
    discrep = IC.ModelDiscrep(name=discrep_model_name, process_space=process_space, measurement_space=measurement_space,
                              coarse_process_model=process_model_coarse, data_model=real_data_model)
    discrep_data = pickle.load(
        open("./saved_data/" + discrepancy_filename, "rb"))
    discrep.discrepancy_dist = discrep_data
    discrep.compute_overall_discrep_model(map_coarse_discrep_to_data_grid=map_coarse_discrep_to_data_grid)

    #update process space based on raw discrep.
    process_space.discrepancies = discrep.discrepancy_dist

    #compute relevant data space quantities.
    measurement_space.bias = discrep.discrepancy_mean_data
    measurement_space.icov = discrep.combined_icov_data
    measurement_space.map_process_to_data_scale = map_coarse_discrep_to_data_grid
#else:
#    discrep = IC.ModelDiscrep(process_space=process_space, measurement_space=measurement_space)

#---create a Bayes model
#use pro_model_coarse for coarse, pro_model_medium for medium
model_name = 'test_krafla_bayes_model'
if load_discrepancy:
    model_name = model_name + '_discrep'

#bmodel_coarse = IC.BayesModel(name=model_name,
#                       process_model=process_model_coarse, 
#                       data_model=real_data_model, 
#                       measurement_space=measurement_space,
#                       parameter_space=parameter_space,
#                       process_space=process_space,
#                       fine_process_model=process_model_fine)
bmodel_coarse = IC.BayesModel(name=model_name,
                       process_model=process_model_coarse, 
                       data_model=real_data_model, 
                       measurement_space=measurement_space,
                       parameter_space=parameter_space,
                       process_space=process_space)

#---test calculate likelihood
start = timeit.default_timer()

ll = bmodel_coarse.lnlike(perm_powers_current=perm_powers_precal)

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

#---test find best fit
f = lambda x: -bmodel_coarse.lnprob(perm_powers_current=x)

#guess = np.tile(-15.0, len(perm_powers_precal))
guess = perm_powers_precal #if trying to improve on John's...


#use opt from last time
#save_path = './saved_data/'
#guess = pickle.load(open(save_path + 'test_krafla_bayes_model_best_fit_solution.p', "rb"))

maxfevs = 1000
fatol = 1e-3
start = timeit.default_timer()

res = minimize(fun=f, x0=guess, method='Nelder-Mead', tol=1e-3,
               options={'disp': True, 'maxfev': maxfevs, 'fatol':fatol})

#res = minimize(fun=f, x0=guess, method='L-BFGS-B', tol=1e-3,
#               options={'disp': True, 'maxfev': maxfevs})

stop = timeit.default_timer()
print('time to find best fit (s): ')
print(stop - start)

best_fit_solution = np.zeros((len(res.x)+1))
best_fit_solution[:-1] = res.x
best_fit_solution[-1] = res.fun

#---save data
save_data = True
if save_data:
    #save best fit data 
    save_path = './saved_data/'
    try:
        os.makedirs(save_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(save_path):
            pass
        else:
            raise
    pickle.dump(best_fit_solution[:-1], open(save_path + bmodel_coarse.name + "_best_fit_solution.p", "wb"))

#---do predictive check
load_data = True
if load_data:
    best_params = pickle.load(open(save_path + bmodel_coarse.name + "_best_fit_solution.p", "rb"))
else:
    best_params = best_fit_solution[:-1]

param_sets = np.array((best_params,))

bmodel_coarse.predictive_checks(parameter_sets=param_sets,subtract_bias=False)
#bmodel_coarse.predictive_checks(parameter_sets=param_sets,subtract_bias=True)

#bmodel_coarse.predictive_checks(parameter_sets=param_sets,subtract_bias=False)
#bmodel_coarse.predictive_checks(parameter_sets=param_sets,subtract_bias=True,do_data_space=False)

#---test emcee
#p_start = best_params
#bmodel.run_emcee(p_start=p_start, 
#                 n_walkers=30, n_burn=10, n_sample=30, save_data=True, run_name='_test_short')

#test find best fit with custom derivative
#f = lambda x: -bmodel.lnprob(perm_powers_current=x)
#df = lambda x: -bmodel.lnprob_grad(perm_powers_current=x)
#
#guess = np.tile(-15.0, 12)
#maxfevs = 200
#fatol = 0.01
#start = timeit.default_timer()
#res = minimize(fun=f, jac=df, x0=guess, method='BFGS', tol=1e-1,
#               options={'disp': True, 'maxfev': maxfevs, 'fatol':fatol})
#
#stop = timeit.default_timer()
#print('time to find best fit with gradient (s): ')
#print(stop - start)
#
#best_fit_solution_wderiv = np.zeros((len(res.x)+1))
#best_fit_solution_wderiv[:-1] = res.x
#best_fit_solution_wderiv[-1] = res.fun


