import numpy as np
from numpy.random import RandomState
import timeit
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pickle
import os
import errno
import re
from scipy import interpolate
from scipy import stats
import emcee
import corner
from functools import reduce
from operator import mul
from itertools import chain
import pandas as pd
from shutil import copyfile
from multiprocessing import Pool
import multiprocessing
import time

#Basic classes:

# ParameterModel
# MeasurementSpaceModel
#
#
#
#
#

class ParameterSpace:
    '''
    Holds basic parameter prior data. Think of ParameterSpace class (i.e. the parameter vector space)
    '''
    def __init__(self, mu=None, sigma=None, resid=None, norm_order=2):

        self.mu = mu
        self.sigma = sigma
        self.norm_order = norm_order
        self.resid = resid

    def compute_parameter_residual(self, perm_powers_current):
        self.resid = perm_powers_current - self.mu

    def choose_random_parameter_subsets(self, parameter_pool, n_subsets):
        return parameter_pool[np.random.randint(0, len(parameter_pool), n_subsets)]


class MeasurementSpace:
    '''
    Holds basic data for 'comparisons' in 'data space'. To be used as input to 
    construct a likelihood function. 
    
    Note: the likelihood function itself is part of a BayesModel.

    Think of the present as MeasurementSpace class (i.e. the data vector space).

    To do - more carefully separate the 'data' and 'process' space comparison elements
    to make more modular.
    '''
    def __init__(self, bias=0.0, bias_function=None, icov=None, sigma=5.0, resid=0.0, map_process_to_data_scale=False):
        
        self.bias = bias
        self.bias_function = bias_function
        self.icov = icov
        self.sigma = sigma
        self.resid = resid
        self.map_process_to_data_scale = map_process_to_data_scale

    #compute_well_residual(process_model=self.process_model,data_model=self.data_model)

    def compute_well_residual(self, process_model=None, data_model=None, n_ignore=1):
        '''
        Assume shape nrows = n wells, ncols = n observations.

        NOTE: Updated to be a dictionary to account for different numbers of observations per well!

        Option to map finer data to coarser model scale. Todo - make an explicit coarse data model?

        '''

        T_obs_all_data = data_model.ss_temps_obs_well
        d_obs_all_data = data_model.d_obs_well
        
        #T_obs_all_data = [x for x in chain(*T_obs_all_data.values())]
        #d_obs_all_data = [x for x in chain(*d_obs_all_data.values())]

        T_obs_all_process = process_model.ss_temps_obs_well
        d_obs_all_process = process_model.d_obs_well

        #T_obs_array_fine_at_coarse = np.zeros((T_obs_array_coarse.shape[0],
        #                                       T_obs_array_coarse.shape[1]-n_ignore))

        #create new coarse data dictionary for process model and take diff with given data
        diffs = {}
        
        #np.zeros((T_obs_array_data.shape[0], T_obs_array_data.shape[1]- n_ignore))

        for i, temp_obs_well_i_process in T_obs_all_process.iteritems():
            d_obs_well_i_data = d_obs_all_data[i]
            d_obs_well_i_process = d_obs_all_process[i]

            temp_obs_well_i_data = T_obs_all_data[i]

            if self.map_process_to_data_scale: 
                T_well_func_process = interpolate.interp1d(d_obs_well_i_process, temp_obs_well_i_process,
                                                    kind='linear', fill_value='extrapolate')
                T_process_at_data_well = T_well_func_process(d_obs_well_i_data)
                diffs[i] = temp_obs_well_i_data[n_ignore:]-T_process_at_data_well[n_ignore:]
            
                #T_well_func_data = interpolate.interp1d(d_obs_well_i_data, temp_obs_well_i_data,
                #                                    kind='linear', fill_value='extrapolate')
                #data should be mapped to nearest

            else: #map data to process
                #model OK to linearly interp. (more regular)
                T_well_func_data = interpolate.interp1d(d_obs_well_i_data, temp_obs_well_i_data,
                                                    kind='nearest', bounds_error=False,fill_value='extrapolate')

                T_data_at_process_well = T_well_func_data(d_obs_well_i_process)
                diffs[i] = T_data_at_process_well[n_ignore:]-temp_obs_well_i_process[n_ignore:]

        self.resid = np.concatenate(diffs.values())

        return

    def map_fine_to_coarse(self, well_number=0, coarse_model=None, fine_model=None):
        
        #T_obs_all_coarse = coarse_model.ss_temps_obs_well
        d_obs_all_coarse = coarse_model.d_obs_well

        T_obs_all_fine = fine_model.ss_temps_obs_well
        d_obs_all_fine = fine_model.d_obs_well


        #T_obs_array_fine_at_coarse = np.zeros((T_obs_array_coarse.shape[0],
        #                                       T_obs_array_coarse.shape[1]-n_ignore))

        d_obs_well_i_coarse = d_obs_all_coarse[well_number]
        d_obs_well_i_fine = d_obs_all_fine[well_number]

        #temp_obs_well_i_coarse = T_obs_all_coarse[well_number]
        temp_obs_well_i_fine = T_obs_all_fine[well_number]

        #print('map')
        #print(d_obs_row_i_coarse)
        #print(d_obs_row_i_fine)
        #print(temp_obs_row_i_coarse)
        #print(temp_obs_row_i_fine)

        T_well_func_fine = interpolate.interp1d(d_obs_well_i_fine, temp_obs_well_i_fine,
                                                kind='nearest', bounds_error=False, fill_value='extrapolate')

        T_fine_at_coarse_well = T_well_func_fine(d_obs_well_i_coarse)

        #print(T_fine_at_coarse_well)

        return (d_obs_well_i_coarse, T_fine_at_coarse_well)
            
        #diffs[i, :] = temp_obs_row_coarse[n_ignore:] - \
        #    T_fine_at_coarse_well[n_ignore:]

    #measurement_space.map_fine_to_coarse(process_model=self.process_model,data_model=self.data_model, well_number=i)

class ProcessSpace:
    '''
    Holds basic data for 'comparisons' in 'process space'. To be used as input to 
    construct a discrepancy model and hence a MeasurementSpace model and and hence a likelihood function. 
    
    Note: the likelihood function itself is part of a BayesModel.

    To do - more carefully separate the 'data' and 'process' space comparison elements
    to make more modular.
    '''

    def __init__(self, discrepancies=0.0, discrepancy_function=None):
        
        self.discrepancies = discrepancies
        self.discrepancy_function = discrepancy_function
        #self.icov = icov
        #self.sigma = sigma
        #self.resid = resid

    def map_fine_to_coarse(self, well_number=0, coarse_model=None, fine_model=None):
        
        #T_obs_all_coarse = coarse_model.ss_temps_obs_well
        d_obs_all_coarse = coarse_model.d_obs_well

        T_obs_all_fine = fine_model.ss_temps_obs_well
        d_obs_all_fine = fine_model.d_obs_well

        #T_obs_array_fine_at_coarse = np.zeros((T_obs_array_coarse.shape[0],
        #                                       T_obs_array_coarse.shape[1]-n_ignore))

        d_obs_well_i_coarse = d_obs_all_coarse[well_number]
        d_obs_well_i_fine = d_obs_all_fine[well_number]

        #temp_obs_well_i_coarse = T_obs_all_coarse[well_number]
        temp_obs_well_i_fine = T_obs_all_fine[well_number]

        #print('map')
        #print(d_obs_row_i_coarse)
        #print(d_obs_row_i_fine)
        #print(temp_obs_row_i_coarse)
        #print(temp_obs_row_i_fine)

        T_well_func_fine = interpolate.interp1d(d_obs_well_i_fine, temp_obs_well_i_fine,
                                                kind='linear', fill_value='extrapolate')

        T_fine_at_coarse_well = T_well_func_fine(d_obs_well_i_coarse)

        #print(T_fine_at_coarse_well)

        return (d_obs_well_i_coarse, T_fine_at_coarse_well)
            
        #diffs[i, :] = temp_obs_row_coarse[n_ignore:] - \
        #    T_fine_at_coarse_well[n_ignore:]
        
    
    def compute_well_residual(self, coarse_model=None, fine_model=None, n_ignore=1):
        '''
        Maps fine to coarse. TODO - add option to 'lift' coarse model to fine?

        '''

        T_obs_all_coarse = coarse_model.ss_temps_obs_well
        d_obs_all_coarse = coarse_model.d_obs_well

        T_obs_all_fine = fine_model.ss_temps_obs_well
        d_obs_all_fine = fine_model.d_obs_well

        #T_obs_array_fine_at_coarse = np.zeros((T_obs_array_coarse.shape[0],
        #                                       T_obs_array_coarse.shape[1]-n_ignore))

        #create new coarse obs array for fine data and take diff
        #diffs = np.zeros((T_obs_array_coarse.shape[0], T_obs_array_coarse.shape[1]- n_ignore))
        diffs = {}

        for i, temp_obs_well_i_fine in T_obs_all_fine.iteritems():
            d_obs_well_i_coarse = d_obs_all_coarse[i]
            d_obs_well_i_fine = d_obs_all_fine[i] 
            temp_obs_well_coarse = T_obs_all_coarse[i]

            T_well_func_fine = interpolate.interp1d(d_obs_well_i_fine, temp_obs_well_i_fine,
                                                    kind='linear', fill_value='extrapolate')
            T_fine_at_coarse_well = T_well_func_fine(d_obs_well_i_coarse)
            diffs[i] = temp_obs_well_coarse[n_ignore:]-T_fine_at_coarse_well[n_ignore:]

        self.resid = np.concatenate(diffs.values())
        return

class ModelDiscrep:
    '''

    Implements model discrepancy calculations, using both ProcessSpace and MeasurementSpace functionality.

    The 'process_space' part is a ProcessSpace model which allows comparisons to be made in this space,
    e.g. residuals to be computed. 

    The goal is to construct a MeasurementSpace model to pass to a BayesModel to compute a likelihood.

    TODO - enforce types?

    '''

    def __init__(self, name='', coarse_process_model=None, fine_process_model=None, data_model=None, process_space=None, measurement_space=None,
                 parameter_set_pool=None, discrepancy_dist=None, discrepancy_dist_data=None, discrepancy_perms=None, discrepancy_mean_data=None, 
                 discrepancy_icov_data=None, combined_icov_data=None, discrepancy_cov_data=None):
        
        self.name = name
        self.coarse_process_model = coarse_process_model
        self.fine_process_model = fine_process_model
        self.data_model = data_model
        self.process_space = process_space
        self.measurement_space = measurement_space
        self.parameter_set_pool = parameter_set_pool
        self.discrepancy_dist = discrepancy_dist
        self.discrepancy_dist_data = discrepancy_dist_data
        self.discrepancy_perms = discrepancy_perms
        self.discrepancy_mean_data = discrepancy_mean_data
        self.discrepancy_icov_data = discrepancy_icov_data
        self.combined_icov_data = combined_icov_data
        self.discrepancy_cov_data = discrepancy_cov_data

    #raw model discrep
    def compute_raw_model_discrepancy(self, num_runs=1, n_ignore=1, save_data=True):
        
        num_candidates = self.parameter_set_pool.shape[0]
        num_params = self.parameter_set_pool.shape[1]

        #dummy simulation for shape of obs.
        i = 0
        while i < 1:
            perm_powers_current = self.parameter_set_pool[np.random.randint(0, num_candidates), :]
            self.coarse_process_model.set_rock_permeabilities(perm_powers_current)
            self.coarse_process_model.simulate()

            if not self.coarse_process_model.is_good_model:
                continue
            i = 1

        #shape_full = self.coarse_process_model.ss_temps_obs_well.shape
        #num_obs_flat = np.product(np.subtract(shape_full, (0, n_ignore)))

        n_total = len(np.hstack(self.coarse_process_model.ss_temps_obs_well.values()))
        n_wells = len(self.coarse_process_model.ss_temps_obs_well.keys())
        num_obs_flat = n_total - n_ignore*n_wells
        
        self.discrepancy_dist = np.zeros((num_runs, num_obs_flat))
        self.discrepancy_perms = np.zeros((num_runs, num_params))

        start = timeit.default_timer()
        i = 40
        while i < num_runs+40:
            # generate new permeability realisation
            #perm_powers_current = np.random.normal(loc=-15.,scale=1.5,size=num_params)
            perm_powers_current = self.parameter_set_pool[np.random.randint(0, num_candidates), :]
 
            # compute coarse simulation
            print("computing coarse model number", str(i))
            self.coarse_process_model.set_rock_permeabilities(perm_powers_current)
            self.coarse_process_model.simulate()
            if not self.coarse_process_model.is_good_model:
                continue
            
            newfn = 'krafla_vcoarse'+str(i)+'.listing'
            path = os.path.join(os.getcwd(), 'input-files', 'coarse', '')
            copyfile(path+'krafla_vcoarse.listing', path+newfn)

            # compute fine simulation
            #HERE
            print("computing fine model number", str(i))
            self.fine_process_model.set_rock_permeabilities(perm_powers_current)
            self.fine_process_model.simulate()
            if not self.fine_process_model.is_good_model:
                continue
            
            newfn = 'krafla_fine'+str(i)+'.listing'
            path = os.path.join(os.getcwd(), 'input-files', 'medium', '')
            copyfile(path+'krafla_fine.listing', path+newfn)
            
            # compute and store. STORE FOR KNOWN PERMS? OR FOR REALISATIONS?
            self.process_space.compute_well_residual(
                coarse_model=self.coarse_process_model, 
                fine_model=self.fine_process_model,
                n_ignore=n_ignore)

            self.discrepancy_dist[i-40, :] = self.process_space.resid
            self.discrepancy_perms[i-40, :] = perm_powers_current

            #update success count
            i = i + 1
        stop = timeit.default_timer()
        print('time to compute discrepancy dist (s): ')
        print(stop - start)

        #save solution data
        if save_data:
            save_path = './saved_data/'
            try:
                os.makedirs(save_path)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(save_path):
                    pass
                else:
                    raise
            pickle.dump(self.discrepancy_dist, open(
                save_path + "discrepancies_" + self.name + ".p", "wb"))

            pickle.dump(self.discrepancy_perms, open(
                save_path + "discrepancy_permeabilities_" + self.name + ".p", "wb"))

    #dealing with outlier discreps
    def remove_and_replace_outlier_discrep(self, z_dev_tol=5.):
        '''
        Remove outliers from each location and replace by median
        '''
        
        for i in range(0,self.discrepancy_dist.shape[1]):
            discreps_at_location = self.discrepancy_dist[:,i]
            
            abs_dev = np.abs(discreps_at_location - np.median(discreps_at_location))
            mad = np.median(abs_dev)

            discreps_at_location[np.divide(abs_dev,mad) > z_dev_tol] = np.median(discreps_at_location)
            self.discrepancy_dist[:,i] = discreps_at_location


    def map_coarse_discrep_to_data_model_grid(self,n_ignore=1):
        
        r = re.compile("OBS*")
        obs_wells_list = filter(r.match, self.coarse_process_model.geom.well.keys())

        #geom_model = self.coarse_process_model.geom

        #initialise new discrep 
        #n_total = len(np.hstack(data_model.ss_temps_obs_well.values()))
        #n_wells = len(self.data_model.ss_temps_obs_well.keys())
        #num_obs_flat = n_total - n_ignore*n_wells
        discrep_at_data_dict = {} #np.zeros((self.discrepancy_dist.shape[0],num_obs_flat))

        d_obs_well_data = self.data_model.d_obs_well
        #print("d_obs_well_data = ", d_obs_well_data)
        #time.sleep(60)
        
        if self.coarse_process_model.ss_temps_obs_well is None:
            self.coarse_process_model.simulate()

        discrep_starting_index = [len(self.coarse_process_model.ss_temps_obs_well[key])-n_ignore for key in np.arange(len(obs_wells_list))]
        discrep_starting_index = [0] + list(np.cumsum(discrep_starting_index))


        for i,welli in enumerate(obs_wells_list):
            #print("well = ", welli, "i = ", i)
            d_obs_well_i_process = self.coarse_process_model.d_obs_well[i][n_ignore:]
            discreps_well_i = self.discrepancy_dist[:,discrep_starting_index[i]:discrep_starting_index[i+1]]

            discrep_func = interpolate.interp1d(d_obs_well_i_process, discreps_well_i,
                                kind='linear', fill_value='extrapolate')
            
            d_obs_well_i_data = d_obs_well_data[i][n_ignore:]
            discrep_at_data_dict[i] = discrep_func(d_obs_well_i_data)
            #print("well = ", welli, " i = ", i, " discrep_at_data_dict[i] = ", discrep_at_data_dict[i] )
            #time.sleep(60)
        self.discrepancy_dist_data = np.hstack(discrep_at_data_dict.values())
        #self.discrep_func = discrep_func
           
        return

    #overall, normal model for discrep: icov and bias/mu
    def compute_overall_discrep_model(self,save_data=True,map_coarse_discrep_to_data_grid=False,n_ignore=1):
        '''
        Function for determining overall inverse covariance. Should be used 'offline' i.e. just once outside of any algorithm.

        Once determined, should save and pass directly to likelihood etc.
        '''

        if map_coarse_discrep_to_data_grid:
            self.map_coarse_discrep_to_data_model_grid(n_ignore=n_ignore)
        else: 
            self.discrepancy_dist_data = self.discrepancy_dist
            
 #       discrep_data = pickle.load(open("./saved_data/" + "discrepancies_test_discrep_krafla_combined_naive.p", "rb"))
#        self.discrepancy_dist_data = discrep_data

        #mean-centre discrepancies - data space
        discreps_mean_data = np.mean(self.discrepancy_dist_data, axis=0)
        discreps_centred_data = self.discrepancy_dist_data - discreps_mean_data

        #determine covariance of discreps
        cov_discreps_data = np.cov(discreps_centred_data.T)

        #add data covariance - assuming diagonal here. Can change in principle.
        if self.measurement_space.icov is not None:
            cov_data = cov_discreps_data + np.linalg.inv(self.measurement_space.icov)
        else:
            sigma_data = self.measurement_space.sigma
            cov_data = cov_discreps_data + np.diag(np.tile(sigma_data**2,discreps_mean_data.shape)) #sigma_data**2

        #invert
        icov_data = np.linalg.inv(cov_data)
        #icov_discreps_data = np.linalg.inv(cov_discreps_data)

        #save icov
        if save_data:
            save_path = './saved_data/'
            try: 
                os.makedirs(save_path)
            except OSError as exc: # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(save_path):
                    pass 
                else: raise
            pickle.dump(icov_data, open(save_path + "icov_"+ self.name + ".p", "wb"))
            pickle.dump(discreps_mean_data, open(save_path + "discreps_mean_" + self.name + ".p", "wb"))

        self.combined_icov_data = icov_data
        #self.discrepancy_icov_data = icov_discreps_data
        self.discrepancy_mean_data = discreps_mean_data
        self.discrepancy_cov_data = cov_discreps_data

# Functions and classes for statistical/inverse methods tools

class BayesModel:
    def __init__(self, name='', process_model=None, data_model=None,fine_process_model=None,
                measurement_space=None, process_space=None,parameter_space=None,
                sampler_flatchain=None):

        self.name = name
        self.process_model = process_model
        self.data_model = data_model
        self.fine_process_model = fine_process_model # for e.g. posterior predictice checks.
        #self.discrepancy_model = discrepancy_model # for e.g. posterior predictice checks.
        self.process_space = process_space
        self.measurement_space = measurement_space
        #self.comparison_model = comparison_model
        self.parameter_space = parameter_space
        self.sampler_flatchain = sampler_flatchain
        return

    def lnlike(self, perm_powers_current):
        #update process model
        #print(perm_powers_current)
        self.process_model.set_rock_permeabilities(perm_powers_current)
        
        self.process_model.simulate()
        
        #compute comparision residuals etc
        #assumes synthetic model has been run...or real data exists
        self.measurement_space.compute_well_residual(process_model=self.process_model,
                                                    data_model=self.data_model)

        #corrections and ll calc.                                          
        #if self.measurement_space.bias_function is not None:
        #    self.measurement_space.bias = self.measurement_space.bias_function(perm_powers_current)
        self.measurement_space.resid = self.measurement_space.resid - self.measurement_space.bias

        #if self.measurement_space.icov is not None:
        #    ll = -np.dot(self.measurement_space.resid, 
        #                 np.dot(self.measurement_space.icov, self.measurement_space.resid)) / 2.0
        #else:
        weighted_resid = np.divide(self.measurement_space.resid, self.measurement_space.sigma)
            #print(weighted_resid)
            #ll = -0.5 * np.power(np.linalg.norm(weighted_resid, axis=0), 2)
        ll = -0.5 * np.power(np.linalg.norm(weighted_resid, axis=0), 2)
        print('ll = ', ll)
        return ll

    def lnprior(self, perm_powers_current):#, sigma, mu=-15,prior_norm_order=2):

        #update parameter residual
        self.parameter_space.compute_parameter_residual(perm_powers_current)

        weighted_resid = np.divide(self.parameter_space.resid, self.parameter_space.sigma)

        #note: here, param = log(perm) ~ normal.
        lp = -0.5 * np.power(np.linalg.norm(weighted_resid, axis=0),self.parameter_space.norm_order)

        return lp

    def lnprob(self,perm_powers_current): #, do_plot=False, is_silent=True):

        lp = self.lnprior(perm_powers_current)

        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(perm_powers_current=perm_powers_current) #, do_plot=do_plot, is_silent=is_silent)

    def lnprob_grad(self,perm_powers_current):
        '''
        Currently slighly inefficient since potentially re-calcs process model a few extra times...
        '''
        #calculate jacobian
        self.process_model.compute_jacobian_for_perm_powers(perm_powers_current, delta=0.01, as_fraction=True)
        
        #update parameter residual
        self.parameter_space.compute_parameter_residual(perm_powers_current)

        #update process model
        self.process_model.set_rock_permeabilities(perm_powers_current)
        self.process_model.simulate()

        return np.dot(self.measurement_space.resid, self.process_model.jacobian) + self.parameter_space.resid

    def run_emcee(self, p_start, n_walkers=300, n_burn=100, n_sample=500, save_data=True, run_name='_test'):
        '''
        implement here
        '''

        num_params = len(p_start)
        n_dim = num_params

        #starting ensemble
        p0 = [np.random.normal(loc=p_start,scale=0.1) for i in range(n_walkers)]

        #create lambda here....

        #pool = Pool(processes=4)
        #with Pool() as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.lnprob)
        t0 = timeit.default_timer()
        p_post_burn, prob_post_burn, state_post_burn = sampler.run_mcmc(p0,n_burn)
        t1 = timeit.default_timer()
        t_burn = t1-t0 
        print('burn time: ')
        print(t_burn)
        
        save_path = './saved_data/'
        pickle.dump(sampler.chain, open(save_path + "sampler_chainburn" + run_name + ".p", "wb"))
        pickle.dump(sampler.flatchain, open(save_path + "sampler_flatchainburn" + run_name + ".p", "wb"))
        pickle.dump(sampler.lnprobability, open(save_path + "sampler_lnprobburn" + run_name + ".p", "wb"))
        pickle.dump(sampler.flatlnprobability, open(save_path + "sampler_flatlnprobburn" + run_name + ".p", "wb"))
        
        sampler.reset()
        t0 = timeit.default_timer()
        p_post_sample, prob, state = sampler.run_mcmc(p_post_burn, n_sample)
        t1 = timeit.default_timer()
        t_sample = t1-t0
        print('sample time: ')
        print(t_sample)

        print('acceptance fraction: ')
        print(sampler.acceptance_fraction)

        self.sampler_flatchain = sampler.flatchain

        #save mcmc data
        save_path = './saved_data/'
        try:
            os.makedirs(save_path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(save_path):
                pass
            else:
                raise
        pickle.dump(sampler.chain, open(save_path + "sampler_chain" + run_name + ".p", "wb"))
        pickle.dump(sampler.flatchain, open(save_path + "sampler_flatchain" + run_name + ".p", "wb"))
        pickle.dump(sampler.lnprobability, open(save_path + "sampler_lnprob" + run_name + ".p", "wb"))
        pickle.dump(sampler.flatlnprobability, open(save_path + "sampler_flatlnprob" + run_name + ".p", "wb"))

        #load with pickle.load(open("./saved_data/sampler_flatchain.p", "rb" ))
        return

    def predictive_checks(self, parameter_sets, subtract_bias=True, n_ignore=0, project_data_to_process=True,
                                                labels=None,ticks=None,xlims=[50,250],line_props=None,alpha=0.1,plot_wells_list=None):
        '''
        parameter sets should have shape (nsets, nparams)

        Need to re-factor to distinguish discrep and bias.
        '''

        #xmin = xlims[0]
        #xmax = xlims[1]
                        
        r = re.compile("OBS*")
        obs_wells_list = filter(r.match, self.process_model.geom.well.keys())
        #obs_wells_list = plot_wells_list
        
        #Create dictionary of dataframes into which model results will be stored with dimensionality [n_wells](n_obs, n_sets)
        t_models = {}
        d_models = {} #can depth be used as indexing? 
        for welli in list(obs_wells_list):
            t_models[welli] = pd.DataFrame(columns=[j for j,param_sets in enumerate(parameter_sets)])
            d_models[welli] = pd.DataFrame(columns=[j for j,param_sets in enumerate(parameter_sets)])
                                
        #Simulate forward model  the number of times given by length of parameter_sets
        for j,param_set in enumerate(parameter_sets):
            print('param set loop:', j)
            self.process_model.set_rock_permeabilities(param_set)
            self.process_model.simulate()
            
            newfn = 'krafla_fine'+str(j+80)+'.listing'
            path = os.path.join(os.getcwd(), 'input-files', 'medium', '')
            copyfile(path+'krafla_fine.listing', path+newfn)
            

            temps_model = self.process_model.ss_temps
            geom_model = self.process_model.geom

            discrep = self.process_space.discrepancies # process discrep. Add to data space?  
            discrep_starting_index = [len(self.process_model.ss_temps_obs_well[key])-n_ignore for key in np.arange(len(obs_wells_list))]
            discrep_starting_index = [0] + list(np.cumsum(discrep_starting_index))
            
            #store model results into dictionary of dataframes, indexed by wells, column is run number
            for i,welli in enumerate(obs_wells_list):
                (d_model, temp_model) = geom_model.well_values(welli, temps_model, elevation=True)
                t_models[welli][j] = temp_model
                d_models[welli][j] = d_model
                
        for i,welli in enumerate(obs_wells_list):
            print('welli = ', welli, ' i = ', i)
            if project_data_to_process:
                #print('projected data to process')
                # assumes fine has already been set up?
                if self.process_model.ss_temps_obs_well is None:
                    self.process_model.simulate()
                    
                #should always be a data model? Should this call measurement_space? TODO.
                (d_true, temp_true) = self.measurement_space.map_fine_to_coarse(coarse_model=self.process_model, 
                           fine_model=self.data_model, well_number=i)
            else:
                #change to just plot obs at wells
                temp_true= self.data_model.ss_temps_obs_well[i]
                d_true = self.data_model.d_obs_well[i]
            
            plt.plot(temp_true[n_ignore:], d_true[n_ignore:], linestyle='-', color='k', marker='o', markersize=2.0, linewidth=1.0)

            for j,param_set in enumerate(parameter_sets):
                if subtract_bias:
                    #print('subtract bias')
                    if np.asarray(self.process_space.discrepancies).shape==():
                        print('problem with discrepancy space?')
                    else:
                        #bias_i = bias[bias_starting_index[i]:bias_starting_index[i+1]]
        
                        discrep_i = discrep[:,discrep_starting_index[i]:discrep_starting_index[i+1]]
                        bias_i = np.mean(discrep_i, axis=0)
                        #print('bias_i = ', bias_i)
                        #print('t_models = ', t_models[welli][j][n_ignore:])
                        if welli in obs_wells_list:
                            #temps_measured = self.data_model.ss_temps_obs_well[i]
                            #d_measured = self.data_model.d_obs_well[i]
                            #print('temps_measured = ', temps_measured)
                            #print('depths_measured = ', d_measured)
                            #plt.plot(temps_measured[n_ignore:], d_measured[n_ignore:],
                            # linestyle='-', color='k', marker='o', markersize=2.0, linewidth=1.0)
                            #plt.plot(temp_model[n_ignore:]-bias_i,d_model[n_ignore:],'-',color='b',alpha=alpha)
                            plt.plot(t_models[welli][j][n_ignore:]-bias_i,d_models[welli][j][n_ignore:],'-',color='b',alpha=alpha) 

                            #print('plot.savefig 2')
                            #plt.show()
                            #input('Press enter to continue')
                else:
                    if welli in obs_wells_list:
                        #plt.plot(temp_model[n_ignore:], d_model[n_ignore:], '-', color='b', alpha=alpha)
                        plt.plot(t_models[welli][j][n_ignore:], d_models[welli][j][n_ignore:], '-', color='b', alpha=alpha) 
                        #plt.savefig(r'C:\Users\samuels\Desktop\krafla_mcmc\new_figures\model_discreps_no_subtract_bias' + str(welli) + '.pdf')
                        #print('plot.savefig 3')

            axis = plt.gca()
            #print(axis)
        
            if labels is not None:
                axis.set_title(labels['title'])
                axis.xaxis.set_label_text(labels['xlabel'])
                axis.yaxis.set_label_text(labels['ylabel'])
                axis.set_xlim(xmin=50)
                axis.set_xlim(xmax=350)
                axis.set_ylim(ymin=500)
                axis.set_ylim(ymax=-2500)
                axis.invert_yaxis()
            
            if ticks is not None:
                xticks = plticker.MultipleLocator(base=ticks['xticks_gap'])
                yticks = plticker.MultipleLocator(base=ticks['yticks_gap'])
                axis.xaxis.set_major_locator(xticks)
                axis.yaxis.set_major_locator(yticks)
                axis.set_xlim(xmin=50)
                axis.set_xlim(xmax=350)
                axis.set_ylim(ymin=500)
                axis.set_ylim(ymax=-2500)
                axis.invert_yaxis()
             
            if subtract_bias:
                plt.savefig(r'C:\Users\samuels\Desktop\krafla_mcmc\new_figures\model_discreps_' + str(welli) + '.pdf')
            else:
                plt.savefig(r'C:\Users\samuels\Desktop\krafla_mcmc7\figures\model_discreps_no_subtract_bias' + str(welli) + '.pdf')
            plt.clf()
                
            
        return
#        
#
#        axis = plt.gca()
#        #print(axis)
#    
#        if labels is not None:
#            axis.set_title(labels['title'])
#            axis.xaxis.set_label_text(labels['xlabel'])
#            axis.yaxis.set_label_text(labels['ylabel'])
#            axis.set_xlim(xmin=xmin)
#            axis.set_xlim(xmax=xmax)
#        
#        if ticks is not None:
#            xticks = plticker.MultipleLocator(base=ticks['xticks_gap'])
#            yticks = plticker.MultipleLocator(base=ticks['yticks_gap'])
#            axis.xaxis.set_major_locator(xticks)
#            axis.yaxis.set_major_locator(yticks)
#            axis.set_xlim(xmin=xmin)
#            axis.set_xlim(xmax=xmax)
#            
#        #print('./figures/model_discreps_' + self.name + '_subtract_bias_' + np.str(subtract_bias) + '.pdf')
#        plt.savefig('./figures/model_discreps_' + self.name + '_subtract_bias_' + np.str(subtract_bias) + '_project_data_' + np.str(project_data_to_process) + '.pdf')
#        plt.show()

    def plot_perm_posterior(self, do_corner_plot=True, do_marginal_plots=False, perm_powers_truths=None, 
                            use_range=True, parameter_range=[-17, -12], nbins=50,
                            show_perm_truths=True,show_marginal_figs=True):

        num_params = len(self.process_model.free_perm_power_values)
        if use_range:
            ranges = [parameter_range for i in range(0, num_params)]
        else:
            ranges = None

        #print(num_params)
        #print(ranges)
        
        labels = []
        for rt in self.process_model.datfile.grid.rocktypelist:
            if self.process_model.islayered:
                labels.append('$k^{'+str(rt)+'}_x$')
                labels.append('$k^{'+str(rt)+'}_y$')
            else:
                labels.append('$k^{'+str(rt)+'}_x$')
                labels.append('$k^{'+str(rt)+'}_y$')
                labels.append('$k^{'+str(rt)+'}_z$')

        if perm_powers_truths is None:
            print('setting truths to middle of parameter range')
            perm_powers_truths = np.repeat(np.mean(parameter_range), num_params)

        if do_corner_plot:
            if show_perm_truths:
                corner_plot_log = corner.corner(self.sampler_flatchain,
                                            truths=perm_powers_truths, range=ranges, bins=nbins,labelpad=100,label_kwargs={"fontsize":20})
            else:
                corner_plot_log = corner.corner(self.sampler_flatchain,
                                                range=ranges, bins=nbins, labelpad=100, label_kwargs={"fontsize": 20})
            corner_plot_log.savefig('./figures/corner_plot_log_' + self.name + '.pdf')
            
        #kernel density estimation for marginals
        if do_marginal_plots:
            
            plt.close()
            for i in range(0, len(perm_powers_truths)):
                x = np.linspace(perm_powers_truths[i]-3, perm_powers_truths[i]+3, 1000)
                my_pdf = stats.gaussian_kde(self.sampler_flatchain[:, i])
                plt.figure()
                print(labels[i], ' max value is', 10**x[np.argsort(my_pdf(x))[-1]] )
                plt.plot(x, my_pdf(x),linewidth=2.0,label='posterior')
                #prior
                plt.plot(x, stats.norm.pdf(x, loc=perm_powers_truths[i], scale=0.75),linewidth=2.0,linestyle='--',label='prior')
                #truths
                if show_perm_truths:
                    plt.vlines(x=perm_powers_truths[i], ymin=0, ymax=np.max(my_pdf(x))+0.1, linestyles='dashed')
                plt.xlabel(labels[i],labelpad=15)
                plt.ylabel('Probability density',labelpad=10)
                plt.legend()
                plt.savefig('./figures/marginal_' + labels[i] + '_' + self.name + '.pdf')

                if show_marginal_figs:
                    plt.show()
                plt.close()


    
    # def find_map(self):
    #    '''
    #    implement here
    #    '''
    #    return [] 

#class ExploreModels:
#    def __init__(self, process_model, data_model, parameter_set):
#        return

    #implement.


#-bayes_model
#lnlike
#lnprior
#lnprob
#mcmc

#-explore
#predictive checks
#visualise mcmc

#-model_discrep
#icov
#compute_all_well_temp_diffs
#compute_model_discrepancy_dist

#-optimise
#best_fit
#profile_sigma
#profile_perm


