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

#---fine synthetic model
synthetic_model_fine = GC.GeoModel(name='test_synthetic_model_fine', 
                                   datfile_name='input-files/elvar-new-tighter-cap/fine-model/2DF002',
                                   incon_name='input-files/elvar-new-tighter-cap/fine-model/2DF002_IC',
                                   geom_name='input-files/elvar-new-tighter-cap/fine-model/g2fine')

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

#---create a parameter model
parameter_model = IC.ParameterModel(mu=-15, sigma=1.5)


#---create or load comparison space model (basis of likelihood function)
load_discrepancy = True
discrepancy_filename = 'discrepancies_combined.p'

#pre-discrep.
comparison_model = IC.ComparisonModel(bias=0.0, sigma=5.0) #without discrep.

if load_discrepancy:
    discrep = IC.ModelDiscrep(comparison_model=comparison_model)
    discrep_data = pickle.load(open("./saved_data/" + discrepancy_filename, "rb"))
    discrep.discrepancy_dist = discrep_data
    #replace outliers before normal approx
    discrep.remove_and_replace_outlier_discrep(z_dev_tol=5.)
    #normal approx
    discrep.compute_normal_discrep_model()
    #update comparison model based on discrep.
    comparison_model.bias = discrep.discrepancy_mean
    comparison_model.icov = discrep.combined_icov 

#---create a Bayes model
#use process_model_coarse for coarse, process_model_medium for medium.
bmodel = IC.BayesModel(name='test_bayes_model_med_fine_tighter_cap_mcmc_with_combined_discrep_no_outliers',
                       process_model=process_model_medium,
                       data_model=synthetic_model_fine,
                       comparison_model=comparison_model,
                       parameter_model=parameter_model)

best_params = pickle.load(open("./saved_data/best_fit_solution.p", "rb"))

#---test emcee
p_start = best_params
bmodel.run_emcee(p_start=p_start,
                 n_walkers=300, n_burn=50, n_sample=500, save_data=True, run_name='_' + bmodel.name)
#bmodel.run_emcee(p_start, n_walkers=30, n_burn=10, n_sample=30, save_data=True, run_name='_test')

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


