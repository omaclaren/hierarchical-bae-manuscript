import matplotlib as mpl
mpl.use("pgf")
import numpy as np
import timeit
import pickle
import os
import errno
from scipy import stats
from lib import GeothermalCore as GC
from lib import InverseCore as IC
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd

#save a default copy of plotting settings
rc_default = plt.rcParams.copy()

#some global plot settings
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
#plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['axes.labelsize'] = 24
#plt.rcParams['axes.textsize'] = 24
plt.rcParams['axes.titlesize'] = 26
plt.rcParams['figure.autolayout'] = 'True'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = 'True'
# along with pgf, uses latex default
plt.rcParams['font.serif'] = ['Computer Modern Roman']
#plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['pgf.preamble'] = [
    "\\usepackage{unicode-math}"]  # unicode math setupr
plt.rcParams['legend.fontsize'] = 14

save_path = './saved_data/'
print('predictive checks')

# tighter cap
perm_powers_truths = np.log10(np.array([5.00000000e-14, 1.00000000e-14, 1.00000000e-16,
                                        1.00000000e-16, 1.00000000e-15, 2.00000000e-14,
                                        2.50000000e-14, 1.00000000e-14, 5.00000000e-16,
                                        5.00000000e-16, 5.00000000e-16, 1.00000000e-14]))

#---coarse process model
#process_model_coarse = GC.GeoModel(name='test_process_model_coarse',
#                                   datfile_name='input-files/elvar-new/coarse-model/2DC002',
#                                   incon_name='input-files/elvar-new/coarse-model/2DC002_IC',
#                                   geom_name='input-files/elvar-new/coarse-model/g2coarse',
#                                   islayered=True)

#---medium process model
process_model_medium = GC.GeoModel(name='test_process_model_medium',
                                   datfile_name='input-files/elvar-new-tighter-cap/medium-model/2DM002',
                                   incon_name='input-files/elvar-new-tighter-cap/medium-model/2DM002_IC',
                                   geom_name='input-files/elvar-new-tighter-cap/medium-model/g2medium',
                                   islayered=True)

#---fine process model
process_model_fine = GC.GeoModel(name='test_process_model_fine',
                                 datfile_name='input-files/elvar-new-tighter-cap/fine-model/2DF002',
                                 incon_name='input-files/elvar-new-tighter-cap/fine-model/2DF002_IC',
                                 geom_name='input-files/elvar-new-tighter-cap/fine-model/g2fine',
                                 islayered=True)

#---fine synthetic model
synthetic_data_model_fine = GC.GeoModel(name='test_synthetic_data_model_fine',
                                   datfile_name='input-files/elvar-new-tighter-cap/fine-model/2DF002',
                                   incon_name='input-files/elvar-new-tighter-cap/fine-model/2DF002_IC',
                                   geom_name='input-files/elvar-new-tighter-cap/fine-model/g2fine',
                                   islayered=True)

# --- data model object
generate_new_data = False
#initialise (then overwrite...) - only needed for process space checks.
synthetic_data_model_fine.set_rock_permeabilities(perm_powers=perm_powers_truths)
synthetic_data_model_fine.simulate()


if generate_new_data:
    synthetic_data_model_fine.generate_synthetic_data(
        perm_powers_truths=perm_powers_truths)
else:
    synthetic_data = pickle.load(open("./saved_data/synthetic_data.p", "rb"))
    synthetic_data_model_fine.ss_temps = synthetic_data['T_measured']
    #syn_model.ss_temps = syn_model.T_measured
    synthetic_data_model_fine.T_noise = synthetic_data['T_noise']
    synthetic_data_model_fine.ss_temps_obs_well = synthetic_data['T_obs_well']
    synthetic_data_model_fine.d_obs_well = synthetic_data['d_obs_well']


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
#                       data_model=synthetic_data_model_fine, 
#                       comparison_model=comparison_model,
#                       parameter_model=parameter_model)

load_discrepancy = True
#load_discrepancy = False
map_coarse_discrep_to_data_grid = False #- default should be False? Instead: map data to process space.
#discrepancy_filename = 'discrepancies_combined_kerinci.p'
discrepancy_filename = 'discrepancies_combined.p'


bmodel_name = 'test_slice_bayes_model'
if load_discrepancy:
    bmodel_name = 'test_slice_bayes_model_discrep'
    discrep = IC.ModelDiscrep(process_space=process_space, measurement_space=measurement_space,
                            coarse_process_model=process_model_medium,data_model=synthetic_data_model_fine)
    #else:
    #    discrep = IC.ModelDiscrep(process_space=process_space, measurement_space=measurement_space)

    discrep_data = pickle.load(open("./saved_data/" + discrepancy_filename, "rb"))
    discrep.discrepancy_dist = discrep_data
    discrep.compute_overall_discrep_model(
        map_coarse_discrep_to_data_grid=map_coarse_discrep_to_data_grid)
        
    #update process space based on raw discrep.
    process_space.discrepancies = discrep.discrepancy_dist

    #compute relevant data space quantities.
    measurement_space.bias = discrep.discrepancy_mean_data
    measurement_space.icov = discrep.combined_icov_data  
    measurement_space.map_process_to_data_scale = map_coarse_discrep_to_data_grid

bmodel_coarse = IC.BayesModel(name=bmodel_name,
                       process_model=process_model_medium, 
                       data_model=synthetic_data_model_fine, 
                       measurement_space=measurement_space,
                       parameter_space=parameter_space,
                       process_space=process_space,
                       fine_process_model=process_model_fine)

#---load mcmc chain and choose subset for predictive plots
#flatchain_filename = 'sampler_flatchain_test_bayes_model_discrep_med_fine_ru_16jan.p'
#flatchain_filename = 'sampler_flatchain_test_slice_bayes_model.p'
if load_discrepancy:
    #flatchain_filename = 'sampler_flatchain_test_slice_bayes_model_discrep_combined.p'
    flatchain_filename = 'sampler_flatchain_test_slice_bayes_model_discrep.p'
    flatchain = pickle.load(open(save_path + flatchain_filename, "rb"))
    bmodel_coarse.sampler_flatchain = flatchain
else:
    flatchain_filename = 'sampler_flatchain_test_slice_bayes_model_no_discrep.p'

#best_params = pickle.load(open(save_path + bmodel_coarse.name + "_best_fit_solution.p", "rb"))
#best_params = pickle.load(
#    open("./saved_data/test_slice_bayes_model_best_fit_solution.p", "rb"))

best_params = perm_powers_truths

#param sets as best or as samples
use_sampled_params = True
if use_sampled_params:
    param_sets = parameter_space.choose_random_parameter_subsets(
        parameter_pool=flatchain, n_subsets=50)
else:
    param_sets = np.array((best_params,))
#

#---test calculate likelihood
start = timeit.default_timer()

ll = bmodel_coarse.lnlike(perm_powers_current=best_params)
#ll = bmodel_coarse.lnlike(perm_powers_current=perm_powers_precal)

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

#bmodel.sampler_flatchain = flatchain

#---predictive checks 
#list_of_obs_wells_to_plot = ['OBS_KRDB1','OBS_LP001','OBS_LP002']
#list_of_obs_wells_to_plot = 
#plot_wells_list = ['OBS_KRDB1', 'OBS_LP001', 'OBS_LP002']
#plot_wells_list = ['OBS_KRDB1']#, 'OBS_LP001', 'OBS_LP002']
#plot_wells_list = ['OBS_LP001']#
#plot_wells_list = ['OBS_LP002'] #, 'OBS_LP001', 'OBS_LP002']
# with bias correction, no re-map
#bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=True, do_data_space=False)
# without bias correction, no re-map
#bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=False, do_data_space=False)

#---predictive checks , no re-map
do_predictive_checks = False
#do_predictive_checks = False 

if do_predictive_checks:

    # with bias correction, no re-map
    labels = {'title': '',
            'xlabel': r'Temperature ($^\circ$C)', r'ylabel': 'Elevation (m)'}
    ticks = {'xticks_gap': 50, 'yticks_gap': 250}

    # with bias correction, with original data, lifted discrepancy
    #bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=True,
    #                                project_data_to_process=False, plot_wells_list=plot_wells_list,
    #                                labels=labels, ticks=ticks)
    # without bias correction, with original data, lifted discrepancy
    #bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=False,
    #                                project_data_to_process=False, plot_wells_list=plot_wells_list,
    #                                labels=labels, ticks=ticks)

    # with bias correction, with projected data
    bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=True,
                                    project_data_to_process=True,
                                    labels=labels, ticks=ticks)
    # without bias correction, with projected data
    bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=False,
                                    project_data_to_process=True,
                                    labels=labels, ticks=ticks)

#do_param_posterior = True
show_perm_truths = True
do_param_marginals = True
do_param_corner = False

if do_param_marginals:
    bmodel_coarse.plot_perm_posterior(do_corner_plot=False, do_marginal_plots=True,
                                      perm_powers_truths=perm_powers_truths, show_perm_truths=show_perm_truths,
                                      show_marginal_figs=False)
if do_param_corner:
    bmodel_coarse.plot_perm_posterior(do_corner_plot=True, do_marginal_plots=False,
                                      perm_powers_truths=perm_powers_truths, nbins=30, show_perm_truths=show_perm_truths)
