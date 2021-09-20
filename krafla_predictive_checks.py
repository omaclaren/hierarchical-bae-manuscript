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
import re

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

perm_powers_precal = np.log10(np.array([
         1e-15,   1e-15,   1e-15,
         2.5e-16, 1.0e-15, 1.0e-15,
         7.5e-16, 1.5e-15, 1.0e-15,
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


save_path = './saved_data/'

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

r = re.compile("OBS*")
obs_wells_list = filter(r.match, process_model_coarse.geom.well.keys())
#for i,welli in enumerate(list_of_obs_wells):
for i,welli in enumerate(obs_wells_list):
    print('welli = ', welli, ' i = ', i)
    df = pd.read_csv('wells/temperature/elevation/' + welli[4:] + '.csv',header=None,sep=' ', error_bad_lines=False)
    df.rename(columns={0:'d',1:'T'},inplace=True) 
    
    real_data_model.d_obs_well[i] = df['d']
    real_data_model.ss_temps_obs_well[i] = df['T']
    
    #real_data_model.d_obs_well[str(welli)] = df['d']
    #real_data_model.ss_temps_obs_well[str(welli)] = df['T']

#---create a basic comparison model (basis of likelihood function)
measurement_space = IC.MeasurementSpace(bias=0.0, sigma=10.0)

#---create a parameter model
parameter_space = IC.ParameterSpace(mu=perm_powers_precal, sigma=0.75)


#-----create a basic process space model
process_space = IC.ProcessSpace()



load_discrepancy = True
map_coarse_discrep_to_data_grid = True #- default should be False? Instead: map data to process space.
discrepancy_filename = 'discrepancies_combined.p'

if load_discrepancy:
    discrep = IC.ModelDiscrep(process_space=process_space, measurement_space=measurement_space,
                              coarse_process_model=process_model_coarse, data_model=real_data_model)
    #else:
    #    discrep = IC.ModelDiscrep(process_space=process_space, measurement_space=measurement_space)

    discrep_data = pickle.load(
        open("./saved_data/" + discrepancy_filename, "rb"))
    discrep.discrepancy_dist = discrep_data
    discrep.remove_and_replace_outlier_discrep()
    discrep.compute_overall_discrep_model(
        map_coarse_discrep_to_data_grid=map_coarse_discrep_to_data_grid)

    #update process space based on raw discrep.
    process_space.discrepancies = discrep.discrepancy_dist

    #compute relevant data space quantities.
    measurement_space.bias = discrep.discrepancy_mean_data
    measurement_space.icov = discrep.combined_icov_data
    measurement_space.map_process_to_data_scale = map_coarse_discrep_to_data_grid

bmodel_name = 'test_krafla_bayes_model_discrep'

#---create a Bayes model
bmodel_coarse = IC.BayesModel(name=bmodel_name,
                       process_model=process_model_fine, 
                       data_model=real_data_model, 
                       measurement_space=measurement_space,
                       parameter_space=parameter_space,
                       process_space=process_space,
                       fine_process_model=process_model_fine)

#---load mcmc chain and choose subset for predictive plots
flatchain_filename = 'sampler_flatchain_test_krafla_bayes_model_discrep_combined.p'#'sampler_flatchain_test_krafla_bayes_model_combined.p'
flatchain = pickle.load(open(save_path + flatchain_filename, "rb"))
bmodel_coarse.sampler_flatchain = flatchain

#param sets as best or as samples
use_sampled_params = True
if use_sampled_params:
    param_sets = parameter_space.choose_random_parameter_subsets(
        parameter_pool=flatchain, n_subsets=15)
#else:
#    param_sets = np.array((best_params,))
#

#---test calculate likelihood
#start = timeit.default_timer()

#ll = bmodel_coarse.lnlike(perm_powers_current=best_params)
#ll = bmodel_coarse.lnlike(perm_powers_current=perm_powers_precal)

#stop = timeit.default_timer()
#print('time to compute lnlike (s): ')
#print(stop - start)

#print('ll:')
#print(ll)

#---test calculate model run time
#start = timeit.default_timer()

#bmodel_coarse.process_model.simulate()

#stop = timeit.default_timer()
#print('time to run model (s): ')
#print(stop - start)

#bmodel.sampler_flatchain = flatchain

#---predictive checks 


plot_wells_list = ["OBS_KJ036",  "OBS_KJ039", "OBS_KJ033", "OBS_KJ030", "OBS_KG010",
                     "OBS_KJ009", "OBS_KJ011", "OBS_KJ020", "OBS_KJ027", "OBS_KJ035",  "OBS_KJ037", 
                     "OBS_KJ040", "OBS_KJ022", "OBS_KJ029",  "OBS_KG003", "OBS_KG005", 
                     "OBS_KG008", "OBS_KJ006",  "OBS_KG012",  "OBS_KJ013", "OBS_KJ014", "OBS_KJ015",
                     "OBS_KJ018", "OBS_KJ019", "OBS_KJ021", "OBS_KJ023", "OBS_KG024", "OBS_KG025", 
                     "OBS_KG026", "OBS_KJ028", "OBS_KJ031",  "OBS_KJ034", "OBS_KW001", 'OBS_KW002',
                     "OBS_IDDP1", "OBS_KJ017" , "OBS_KV001"]#"OBS_KJ038","OBS_KS001",
#['OBS_KRDB1', 'OBS_LP001', 'OBS_LP002']
#plot_wells_list = ['OBS_KRDB1']#, 'OBS_LP001', 'OBS_LP002']
#plot_wells_list = ['OBS_LP001']#
#plot_wells_list = ['OBS_LP002'] #, 'OBS_LP001', 'OBS_LP002']
# with bias correction, no re-map
#bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=True, do_data_space=False)
# without bias correction, no re-map
#bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=False, do_data_space=False)

#---predictive checks , no re-map
#do_predictive_checks = True
do_predictive_checks = True 

if do_predictive_checks:

    # with bias correction, no re-map
    labels = {'title': '',
            'xlabel': r'Temperature ($^\circ$C)', r'ylabel': 'Elevation (m)'}
    ticks = {'xticks_gap': 50, 'yticks_gap': 250}

    # with bias correction, with original data, lifted discrepancy
    bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=False,
                                    project_data_to_process=False, plot_wells_list=plot_wells_list,
                                    labels=labels, ticks=ticks)
    # without bias correction, with original data, lifted discrepancy
    #bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=False,
    #                                project_data_to_process=False, plot_wells_list=plot_wells_list,
    #                                labels=labels, ticks=ticks)

    # with bias correction, with projected data
    #bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=True,
    #                                project_data_to_process=True, plot_wells_list=plot_wells_list,
    #                                labels=labels, ticks=ticks)
    # without bias correction, with projected data
    #bmodel_coarse.predictive_checks(parameter_sets=param_sets, subtract_bias=False,
    #                                project_data_to_process=True, plot_wells_list=plot_wells_list,
    #                                labels=labels, ticks=ticks)

#do_param_posterior = True
show_perm_truths = False
do_param_marginals = False
#do_param_corner = True
#
if do_param_marginals:
    bmodel_coarse.plot_perm_posterior(do_corner_plot=False, do_marginal_plots=True,
                                      perm_powers_truths=perm_powers_precal, show_perm_truths=show_perm_truths,
                                      show_marginal_figs=False)
#if do_param_corner:
#    bmodel_coarse.plot_perm_posterior(do_corner_plot=True, do_marginal_plots=False,
#                                  perm_powers_truths=best_params, nbins=30, show_perm_truths=show_perm_truths)
