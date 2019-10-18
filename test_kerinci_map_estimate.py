import numpy as np
import timeit
import pickle
import os
import errno
from lib import GeothermalCore as GC
from lib import InverseCore as IC
from scipy.optimize import minimize
import pandas as pd

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

    df = pd.read_csv('./saved_data/kerinci_data/Temp_' + welli + '.dat',header=None,sep=' ')
    df.rename(columns={1:'d',0:'T'},inplace=True) 

    real_data_model.d_obs_well[i] = df['d']
    real_data_model.ss_temps_obs_well[i] = df['T']

#---create a basic comparison model (basis of likelihood function)
measurement_space = IC.MeasurementSpace(bias=0.0, sigma=10.0)

#---create a parameter model
parameter_space = IC.ParameterSpace(mu=-15, sigma=1.5)


#-----create a basic process space model
process_space = IC.ProcessSpace()

#-----set up discrepancy info
load_discrepancy = True
map_coarse_discrep_to_data_grid = False  # - default should be False?
#discrepancy_filename = 'discrepancies_combined_kerinci.p'
discrepancy_filename = 'discrepancies_combined_kerinci_map_data.p'

if load_discrepancy:
    discrep = IC.ModelDiscrep(process_space=process_space, measurement_space=measurement_space,
                              coarse_process_model=process_model_coarse, data_model=real_data_model)
    #else:
    #    discrep = IC.ModelDiscrep(process_space=process_space, measurement_space=measurement_space)

    discrep_data = pickle.load(
        open("./saved_data/" + discrepancy_filename, "rb"))
    discrep.discrepancy_dist = discrep_data
    discrep.compute_overall_discrep_model(
        map_coarse_discrep_to_data_grid=map_coarse_discrep_to_data_grid)

    #update process space based on raw discrep.
    process_space.discrepancies = discrep.discrepancy_dist

    #compute relevant data space quantities.
    measurement_space.bias = discrep.discrepancy_mean_data
    measurement_space.icov = discrep.combined_icov_data
    measurement_space.map_process_to_data_scale = map_coarse_discrep_to_data_grid

#---create a Bayes model
#use pro_model_coarse for coarse, pro_model_medium for medium
#bmodel = IC.BayesModel(name='test_bayes_model',
#                       process_model=process_model_medium, 
#                       data_model=synthetic_model_fine, 
#                       comparison_model=comparison_model,
#                       parameter_model=parameter_model)

model_name = 'test_kerinci_bayes_model'
if load_discrepancy:
    model_name = model_name + '_discrep'

bmodel_coarse = IC.BayesModel(name=model_name,
                       process_model=process_model_coarse, 
                       data_model=real_data_model, 
                       measurement_space=measurement_space,
                       parameter_space=parameter_space,
                       process_space=process_space,
                       fine_process_model=process_model_fine)

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
#guess = perm_powers_precal #if trying to improve on John's...
#use opt from last time
save_path = './saved_data/'
guess = pickle.load(open(save_path + 'test_kerinci_bayes_model_best_fit_solution.p', "rb"))

maxfevs = 10000#1000
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

bmodel_coarse.predictive_checks(parameter_sets=param_sets,subtract_bias=False,do_data_space=True)
bmodel_coarse.predictive_checks(parameter_sets=param_sets,subtract_bias=True,do_data_space=True)

bmodel_coarse.predictive_checks(parameter_sets=param_sets,subtract_bias=False,do_data_space=False)
bmodel_coarse.predictive_checks(parameter_sets=param_sets,subtract_bias=True,do_data_space=False)

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


