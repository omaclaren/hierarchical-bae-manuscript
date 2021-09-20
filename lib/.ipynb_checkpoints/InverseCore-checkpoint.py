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

class ParameterModel:
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

# Functions and classes for statistical/inverse methods tools
class ComparisonModel:
    '''
    Holds basic data for a 'comparison' data model i.e. likelihood function.
    Think of as DataSpace class (i.e. the data vector space)
    '''
    def __init__(self, bias=0.0, bias_function=None, icov=None, sigma=5.0, resid=0.0):
        
        self.bias = bias
        self.bias_function = bias_function
        self.icov = icov
        self.sigma = sigma
        self.resid = resid

    def map_fine_to_coarse(self, well_number=0, coarse_model=None, fine_model=None):
        
        T_obs_array_coarse = coarse_model.ss_temps_obs_well
        d_obs_array_coarse = coarse_model.d_obs_well

        T_obs_array_fine = fine_model.ss_temps_obs_well
        d_obs_array_fine = fine_model.d_obs_well

        #T_obs_array_fine_at_coarse = np.zeros((T_obs_array_coarse.shape[0],
        #                                       T_obs_array_coarse.shape[1]-n_ignore))

        d_obs_row_i_coarse = d_obs_array_coarse[well_number, :]
        d_obs_row_i_fine = d_obs_array_fine[well_number, :]

        temp_obs_row_i_coarse = T_obs_array_coarse[well_number, :]
        temp_obs_row_i_fine = T_obs_array_fine[well_number, :]

        #print('map')
        #print(d_obs_row_i_coarse)
        #print(d_obs_row_i_fine)
        #print(temp_obs_row_i_coarse)
        #print(temp_obs_row_i_fine)

        T_well_func_fine = interpolate.interp1d(d_obs_row_i_fine, temp_obs_row_i_fine,
                                                kind='linear', fill_value='extrapolate')

        T_fine_at_coarse_well = T_well_func_fine(d_obs_row_i_coarse)

        #print(T_fine_at_coarse_well)

        return (d_obs_row_i_coarse, T_fine_at_coarse_well)
            
        #diffs[i, :] = temp_obs_row_coarse[n_ignore:] - \
        #    T_fine_at_coarse_well[n_ignore:]
        
    
    def compute_well_residual(self, coarse_model=None, fine_model=None, n_ignore=1):
        '''
        Assume shape nrows = n wells, ncols = n observations.

        '''

        T_obs_array_coarse = coarse_model.ss_temps_obs_well
        d_obs_array_coarse = coarse_model.d_obs_well

        T_obs_array_fine = fine_model.ss_temps_obs_well
        d_obs_array_fine = fine_model.d_obs_well

        #T_obs_array_fine_at_coarse = np.zeros((T_obs_array_coarse.shape[0],
        #                                       T_obs_array_coarse.shape[1]-n_ignore))

        #create new coarse obs array for fine data and take diff
        diffs = np.zeros((T_obs_array_coarse.shape[0], T_obs_array_coarse.shape[1]- n_ignore))
        for i, temp_obs_row_i_fine in enumerate(T_obs_array_fine):
            d_obs_row_i_coarse = d_obs_array_coarse[i, :]
            d_obs_row_i_fine = d_obs_array_fine[i, :] 
            temp_obs_row_coarse = T_obs_array_coarse[i, :]

            T_well_func_fine = interpolate.interp1d(d_obs_row_i_fine, temp_obs_row_i_fine,
                                                    kind='linear', fill_value='extrapolate')
            T_fine_at_coarse_well = T_well_func_fine(d_obs_row_i_coarse)
            diffs[i, :] = temp_obs_row_coarse[n_ignore:]-T_fine_at_coarse_well[n_ignore:]

        self.resid = diffs.flatten()

        return

class BayesModel:
    def __init__(self, name='', process_model=None, data_model=None,
                 comparison_model=None, parameter_model=None,
                 sampler_flatchain=None):

        self.name = name
        self.process_model = process_model
        self.data_model = data_model
        self.comparison_model = comparison_model
        self.parameter_model = parameter_model
        self.sampler_flatchain = sampler_flatchain
        return

    def lnlike(self, perm_powers_current):
        #update process model
        self.process_model.set_rock_permeabilities(perm_powers_current)
        self.process_model.simulate()

        #compute comparision residuals etc
        #assumes synthetic model has been run...
        self.comparison_model.compute_well_residual(coarse_model=self.process_model,
                                                    fine_model=self.data_model)

        #corrections and ll calc.                                          
        if self.comparison_model.bias_function is not None:
            self.comparison_model.bias = self.comparison_model.bias_function(perm_powers_current)
        self.comparison_model.resid = self.comparison_model.resid - self.comparison_model.bias

        if self.comparison_model.icov is not None:
            ll = -np.dot(self.comparison_model.resid, 
                         np.dot(self.comparison_model.icov, self.comparison_model.resid)) / 2.0
        else:
            #pure data without model discrep
            weighted_resid = np.divide(self.comparison_model.resid, self.comparison_model.sigma)
            #ll = -0.5 * np.power(np.linalg.norm(weighted_resid, axis=0), 2)
            ll = -0.5 * np.power(np.linalg.norm(weighted_resid, axis=0), 2)
        return ll

    def lnprior(self, perm_powers_current):#, sigma, mu=-15,prior_norm_order=2):

        #update parameter residual
        self.parameter_model.compute_parameter_residual(perm_powers_current)

        weighted_resid = np.divide(self.parameter_model.resid, self.parameter_model.sigma)

        #note: here, param = log(perm) ~ normal.
        lp = -0.5 * np.power(np.linalg.norm(weighted_resid, axis=0),self.parameter_model.norm_order)

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
        self.parameter_model.compute_parameter_residual(perm_powers_current)

        #update process model
        self.process_model.set_rock_permeabilities(perm_powers_current)
        self.process_model.simulate()

        return np.dot(self.comparison_model.resid, self.process_model.jacobian) + self.parameter_model.resid

    def run_emcee(self, p_start, n_walkers=300, n_burn=100, n_sample=500, save_data=True, run_name='_test'):
        '''
        implement here
        '''

        num_params = len(p_start)
        n_dim = num_params

        #starting ensemble
        p0 = [np.random.normal(loc=p_start,scale=0.1) for i in range(n_walkers)]

        #create lambda here....

        sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.lnprob)
                                                                        
        t0 = timeit.default_timer()
        p_post_burn, prob_post_burn, state_post_burn = sampler.run_mcmc(p0,n_burn)
        t1 = timeit.default_timer()
        t_burn = t1-t0 
        print('burn time: ')
        print(t_burn)

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

        #load with pickle.load(open("./saved_data/sampler_flatchain.p", "rb" ))
        return

    def predictive_checks(self, parameter_sets, subtract_bias=True, n_ignore=1, do_map_fine_to_coarse=False,labels=None,ticks=None,line_props=None):
        '''
        parameter sets should have shape (nsets, nparams)
        '''
        
        
        #if line_props is not None:
        #    plt.setp(current_plot, linewidth=line_props['linewidth'], color=line_props['color'])
        
        #should make part of object...
        r = re.compile("OBS*")
        obs_wells_list = filter(r.match, self.process_model.geom.well.keys())

        #first plot true data
        temps_measured = self.data_model.ss_temps
        geom_measured = self.data_model.geom

        for i,welli in enumerate(obs_wells_list):
            if do_map_fine_to_coarse:
                (d_true, temp_true) = self.comparison_model.map_fine_to_coarse(coarse_model=self.process_model,
                                                                               fine_model=self.data_model, 
                                                                               well_number=i)
                #print(d_true)
                #print(temp_true)
                plt.plot(temp_true[n_ignore:],d_true[n_ignore:],'-',color='k',linewidth=1.0) 
            else:
                (d_true, temp_true) = geom_measured.well_values(welli, temps_measured, elevation=True)
                plt.plot(temp_true[n_ignore:],d_true[n_ignore:],'-',color='k',linewidth=1.0)

        for param_set in parameter_sets:
            self.process_model.set_rock_permeabilities(param_set)
            self.process_model.simulate()

            temps_model = self.process_model.ss_temps
            geom_model = self.process_model.geom

            bias_function = self.comparison_model.bias_function
            bias = self.comparison_model.bias

            for i,welli in enumerate(obs_wells_list):
                (d_model, temp_model) = geom_model.well_values(welli, temps_model, elevation=True)

                if bias_function is not None:
                    bias = bias_function(param_set)

                if subtract_bias:
                    bias = bias.reshape(len(obs_wells_list),-1)
                    plt.plot(temp_model[n_ignore:]-bias[i,:],d_model[n_ignore:],'-',color='b',alpha=0.05) 
                else:
                    plt.plot(temp_model[n_ignore:], d_model[n_ignore:], '-', color='b', alpha=0.05) 
                    
        axis = plt.gca()
        #print(axis)
    
        if labels is not None:
            axis.set_title(labels['title'])
            axis.xaxis.set_label_text(labels['xlabel'])
            axis.yaxis.set_label_text(labels['ylabel'])
        
        if ticks is not None:
            xticks = plticker.MultipleLocator(base=ticks['xticks_gap'])
            yticks = plticker.MultipleLocator(base=ticks['yticks_gap'])
            axis.xaxis.set_major_locator(xticks)
            axis.yaxis.set_major_locator(yticks)
            
        #print('./figures/model_discreps_' + self.name + '_subtract_bias_' + np.str(subtract_bias) + '.pdf')
        plt.savefig('./figures/model_discreps_' + self.name + '_subtract_bias_' + np.str(subtract_bias) + '_do_remap_' + np.str(do_map_fine_to_coarse) + '.pdf')
        plt.show()

    def plot_perm_posterior(self, do_corner_plot=True, do_marginal_plots=False, perm_powers_truths=None, 
                            use_range=True, parameter_range=[-17, -12], nbins=50):

        num_params = len(self.process_model.get_free_rock_permeabilities())
        if use_range:
            ranges = [parameter_range for i in range(0, num_params)]
        else:
            ranges = None

        #print(num_params)
        #print(ranges)

        if perm_powers_truths is None:
            print('setting truths to middle of parameter range')
            perm_powers_truths = np.repeat(np.mean(parameter_range), num_params)

        if do_corner_plot:
            corner_plot_log = corner.corner(self.sampler_flatchain,
                                            truths=perm_powers_truths, range=ranges, bins=nbins)
            corner_plot_log.savefig('corner_plot_log_' + self.name + '.pdf')
            
        #kernel density estimation for marginals
        if do_marginal_plots:
            x = np.linspace(-10, -20, 1000)
            plt.close()
            plt.figure()
            for i in range(0, len(perm_powers_truths)):
                my_pdf = stats.gaussian_kde(self.sampler_flatchain[:, i])
                plt.plot(x, my_pdf(x))
                #prior
                plt.plot(x, stats.norm.pdf(x, loc=-15, scale=1.5))
                #truths
                plt.vlines(x=perm_powers_truths[i], ymin=0, ymax=np.max(my_pdf(x))+0.1, linestyles='dashed')
                plt.show()


    
    # def find_map(self):
    #    '''
    #    implement here
    #    '''
    #    return [] 


class ModelDiscrep:
    def __init__(self, name='', coarse_process_model=None, fine_process_model=None, comparison_model=None,
                 parameter_set_pool=None, discrepancy_dist=None, discrepancy_perms=None,
                 discrepancy_mean=None, discrepancy_icov=None, combined_icov=None, discrepancy_cov=None):
        
        self.name = name
        self.coarse_process_model = coarse_process_model
        self.fine_process_model = fine_process_model
        self.comparison_model = comparison_model
        self.parameter_set_pool = parameter_set_pool
        self.discrepancy_dist = discrepancy_dist
        self.discrepancy_perms = discrepancy_perms
        self.discrepancy_mean = discrepancy_mean
        self.discrepancy_icov = discrepancy_icov
        self.combined_icov = combined_icov
        self.discrepancy_cov = discrepancy_cov

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

        shape_full = self.coarse_process_model.ss_temps_obs_well.shape
        num_obs_flat = np.product(np.subtract(shape_full, (0, n_ignore)))
        
        self.discrepancy_dist = np.zeros((num_runs, num_obs_flat))
        self.discrepancy_perms = np.zeros((num_runs, num_params))

        start = timeit.default_timer()
        i = 0
        while i < num_runs:
            # generate new permeability realisation
            #perm_powers_current = np.random.normal(loc=-15.,scale=1.5,size=num_params)
            perm_powers_current = self.parameter_set_pool[np.random.randint(0, num_candidates), :]

            # compute coarse simulation
            self.coarse_process_model.set_rock_permeabilities(perm_powers_current)
            self.coarse_process_model.simulate()
            if not self.coarse_process_model.is_good_model:
                continue

            # compute fine simulation
            #HERE
            self.fine_process_model.set_rock_permeabilities(perm_powers_current)
            self.fine_process_model.simulate()
            if not self.fine_process_model.is_good_model:
                continue

            # compute and store. STORE FOR KNOWN PERMS? OR FOR REALISATIONS?
            self.comparison_model.compute_well_residual(
                coarse_model=self.coarse_process_model, 
                fine_model=self.fine_process_model,
                n_ignore=n_ignore)

            self.discrepancy_dist[i, :] = self.comparison_model.resid
            self.discrepancy_perms[i, :] = perm_powers_current

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


    #normal model discrep: icov and bias/mu
    def compute_normal_discrep_model(self,save_data=True):
        '''
        Function for determining inverse covariance. Should be used 'offline' i.e. just once outside of any algorithm.

        Once determined, should save and pass directly to likelihood etc.
        '''
        #mean-centre discrepancies
        discreps_mean = np.mean(self.discrepancy_dist, axis=0)
        discreps_centred = self.discrepancy_dist - discreps_mean

        #determine covariance of discreps
        cov_discreps = np.cov(discreps_centred.T)

        #add data covariance
        sigma_data = self.comparison_model.sigma
        cov = cov_discreps + np.diag(np.tile(sigma_data**2,discreps_mean.shape)) #sigma_data**2

        #invert
        icov = np.linalg.inv(cov)
        icov_discreps = np.linalg.inv(cov_discreps)

        #save icov
        if save_data:
            save_path = './saved_data/'
            try: 
                os.makedirs(save_path)
            except OSError as exc: # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(save_path):
                    pass 
                else: raise
            pickle.dump(icov, open(save_path + "icov_"+ self.name + ".p", "wb"))
            pickle.dump(discreps_mean, open(save_path + "discreps_mean_" + self.name + ".p", "wb"))

        self.combined_icov = icov
        self.discrepancy_icov = icov_discreps
        self.discrepancy_mean = discreps_mean
        self.discrepancy_cov = cov_discreps

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


