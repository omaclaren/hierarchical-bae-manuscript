import numpy as np
import timeit
import pickle
import os
import errno
from lib import GeothermalCore as GC
from lib import InverseCore as IC
from scipy.optimize import minimize

#---truth
#perm_powers_truths = np.log10(np.array([5.00000000e-14, 1.00000000e-14, 1.00000000e-16, 
#                                        1.00000000e-16, 1.00000000e-15, 2.00000000e-14, 
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
generate_new_data = True 
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
bmodel = IC.BayesModel(name='test_bayes_model',
                       process_model=process_model_medium,
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

#---test find best fit
f = lambda x: -bmodel.lnprob(perm_powers_current=x)
guess = np.tile(-15.0, 12)
maxfevs = 1000
fatol = 1e-3
start = timeit.default_timer()

res = minimize(fun=f, x0=guess, method='Nelder-Mead', tol=1e-3,
               options={'disp': True, 'maxfev': maxfevs, 'fatol':fatol})

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
    pickle.dump(best_fit_solution[:-1], open(save_path + "best_fit_solution.p", "wb"))

#---do predictive check
load_data = True
if load_data:
    best_params = pickle.load(open("./saved_data/best_fit_solution.p", "rb"))
else:
    best_params = best_fit_solution[:-1]

param_sets = np.array((best_params,))
bmodel.predictive_checks(parameter_sets=param_sets)

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


