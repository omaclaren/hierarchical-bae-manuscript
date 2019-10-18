import numpy as np
from numpy.random import RandomState
import timeit
import matplotlib.pyplot as plt
import pickle
import os
import errno
import re
import platform
from collections import OrderedDict
from t2data import *
from t2listing import *
import copy

class GeoModel:
    '''
    Class for holding geothermal simulation models, esp. autough2 models.
    Can also hold pure data models?
    '''
    def __init__(self, name=None, datfile_name=None, incon_name=None, geom_name=None, 
                results=None, is_good_model=False, ss_temps=None, ss_temps_obs_well=None, 
                d_obs_well=None, jacobian=None, T_noise=None, islayered=False,
                rocktype_to_perm_power_index=None,free_perm_power_values=None):

        self.name = name
        self.datfile_name = datfile_name
        self.incon_name = incon_name
        self.geom_name = geom_name
        self.results = results
        self.is_good_model = is_good_model
        self.ss_temps = ss_temps
        self.ss_temps_obs_well = ss_temps_obs_well
        self.d_obs_well = d_obs_well
        self.jacobian = jacobian
        #self.T_measured = T_measured
        self.T_noise = T_noise 
        # add actual datfile and geom to model
        self.datfile = t2data(self.datfile_name + '.dat')
        self.geom = mulgrid(self.geom_name + '.dat')
        self.islayered = islayered
        self.rocktype_to_perm_power_index = self.construct_rocktype_to_perm_index()
        self.free_perm_power_values = free_perm_power_values

    def set_rock_permeabilities(self, perm_powers):
        '''
        self updates:
            datfile -> datfile (permeabilities)
        '''
        #datfile = t2data(self.datfile_name + '.dat')

        #xy permeabilities are the same
        if self.islayered:
            for i, rt in enumerate(self.datfile.grid.rocktypelist):
                rt.permeability[0:2] = np.power(10, perm_powers[2 * i])
                rt.permeability[2] = np.power(10, perm_powers[2 * i + 1])

        else:
            #need to use perm_powers_dict
            #raise Exception('Non-layered models not implemented yet')
            for i, rt in enumerate(self.datfile.grid.rocktypelist):
                #dict needs to return indices of x,y,z perm in perm powers, given rock type. Must be set.
                rt.permeability = np.power(10,perm_powers[self.rocktype_to_perm_power_index[str(rt)]])

        self.free_perm_power_values = np.copy(perm_powers) #store copy of free perm_power values. These are params for est.
        
        #need to write and reload?? Why?? 
        self.datfile.write(self.datfile_name + '.dat')
        self.datfile = t2data(self.datfile_name + '.dat')

    def construct_rocktype_to_perm_index(self):
        '''
        Only actually needed if not assuming layered, but will construct anyway by default.

        Here assumes each rocktype has own x,y,z permeability.

        Necessary to relate model to external user input in form of permeability array.

        These enable a full permeability array to be built from an array of free permeabilities.
        '''

        ppi = {}
        if self.islayered:
            for i, rt in enumerate(self.datfile.grid.rocktypelist):
                #self.rocktype_to_perm_power_index[str(rt)] = np.array(
                ppi[str(rt)] = np.array([int(2 * i), int(2 * i), int(2 * i) + 1])
        else:
            #print('Using fully layered since only layered or fully unlayered implemented. Todo - intermediate cases.')
            #self.rocktype_to_perm_power_index = {}
            for i, rt in enumerate(self.datfile.grid.rocktypelist):
                #self.rocktype_to_perm_power_index[str(rt)] = np.array(
                ppi[str(rt)] = np.array([int(3 * i), int(3 * i) + 1, int(3 * i) + 2])

        return ppi

    def get_all_rock_permeabilities(self):
        perm_array = np.asarray([rt.permeability for rt in self.datfile.grid.rocktypelist])
        return perm_array.flatten()

    # def get_free_rock_permeabilities(self):
    #     '''
    #     Updated to use mapping? Only returns one value for any duplicated permeability. 
    #     E.g. if x and y perms are the same, just returns e.g. [px,pz] not [px,py,pz] etc.

    #     WARNING: INCORRECT AND UNNEEDED!
    #     '''
    #     perm_array_all = self.get_all_rock_permeabilities()

    #     unique_indices = []
    #     #the following removes repeated indices (for each rock type) in the rocktype_to_perm_power_index
    #     for rt in self.datfile.grid.rocktypelist:
    #         unique_indices.extend(list(OrderedDict.fromkeys((self.rocktype_to_perm_power_index[str(rt)]))))

    #     perm_array = np.array(perm_array_all[unique_indices])

    #     return perm_array.flatten()

    def add_fixed_wells(self, save_geom=False):
        '''
        self updates:
            geom -> geom
        Currently fixed wells to add.
        '''

        well1 = well('OBS 1', [np.array([100, 0.5, 0]),np.array([100, 0.5, -1510])])
        well2 = well('OBS 2', [np.array([400, 0.5, 0]),np.array([400, 0.5, -1510])])
        well3 = well('OBS 3', [np.array([700, 0.5, 0]),np.array([700, 0.5, -1510])])
        well4 = well('OBS 4', [np.array([1000, 0.5, 0]),np.array([1000, 0.5, -1510])])
        well5 = well('OBS 5', [np.array([1300, 0.5, 0]),np.array([1300, 0.5, -1510])])
        well6 = well('OBS 6', [np.array([1600, 0.5, 0]),np.array([1600, 0.5, -1510])])
        well7 = well('OBS 7', [np.array([1900, 0.5, 0]),np.array([1900, 0.5, -1510])])

        self.geom.add_well(well1)
        self.geom.add_well(well2)
        self.geom.add_well(well3)
        self.geom.add_well(well4)
        self.geom.add_well(well5)
        self.geom.add_well(well6)
        self.geom.add_well(well7)

        if save_geom:
            self.geom.write()

    def rename_wells_as_obs(self, well_list_for_rename, delete_old=False, save_geom=False):

        '''
        To be used when you have existing wells and want to make them the targets for matching observations.

        Renames these so they fit common format.
        '''

        for welli in well_list_for_rename:
            obs_name = 'OBS_' + welli
            self.geom.well[obs_name] = self.geom.well[welli]
            self.geom.well[obs_name].name = obs_name
            
            if delete_old:
                del self.geom.well[welli]

        if save_geom:
            self.geom.write()

    def update_obs_well_temps(self):
        
        try:
            r = re.compile("OBS*")
            obs_wells_list = filter(r.match, self.geom.well.keys())
        except AttributeError:
            print('No observation wells specified: adding manually')
            obs_wells_list = ['OBS 1', 'OBS 2', 'OBS 3',
                              'OBS 4', 'OBS 5', 'OBS 6', 'OBS 7']

        (d_well_0, T_well_0) = self.geom.well_values(obs_wells_list[0], 
                                                     self.ss_temps, elevation=True)

        #ignore first n_ignore entries in fitting.
        #all_obs_temps = np.zeros((len(obs_wells_list), len(T_well_0)))
        #all_obs_d = np.zeros((len(obs_wells_list), len(d_well_0)))

        all_obs_temps = {}
        all_obs_d = {}

        for i, welli in enumerate(obs_wells_list):
            (d_well, T_well) = self.geom.well_values(welli, self.ss_temps, elevation=True)
            all_obs_temps[i] = T_well
            all_obs_d[i] = d_well

        self.ss_temps_obs_well = copy.deepcopy(all_obs_temps)
        self.d_obs_well = copy.deepcopy(all_obs_d)

    def simulate(self, is_silent=True, reshape=False, do_update_obs_wells=True):
        '''
        self updates:
            results -> results
            ss_temps -> ss_temps
            ss_temps_obs_well -> ss_temps_obs_well 
            d_obs_well -> d_obs_well
            is_good_model -> is_good_model
        '''
        self.is_good_model = True
        #datfile = t2data(self.datfile_name + '.dat')
        #geom = mulgrid(self.geom_name + '.dat')
        
        if platform.system() =='Linux':

            self.datfile.run(save_filename='output-files/saved_run',
                    incon_filename=self.incon_name, simulator='AUTOUGH2_42D', silent=is_silent)
        elif platform.system() == 'Darwin':
            self.datfile.run(save_filename='output-files/saved_run',
                incon_filename=self.incon_name, simulator='AUTOUGH2_MAC', silent=is_silent)
        else:
            print('UNRECOGNISED SIMULATOR SYSTEM')
            #results = t2listing(datfile_name+'.listing')
            self.is_good_model = False
            self.ss_temps = np.zeros(int(self.geom.num_layers * self.geom.num_columns))

        try:
            #might need to make sure I delete edit file each time...
            self.results = t2listing(self.datfile_name + '.listing')
            self.results.last
            self.ss_temps = self.results.element['Temperature']
        except:
            print('BAD MODEL')
            #results = t2listing(datfile_name+'.listing')
            self.is_good_model = False
            self.ss_temps = np.zeros(int(self.geom.num_layers * self.geom.num_columns))

        #might be unnecessary again here! Just flatten after!
        if reshape:
            self.ss_temps = self.ss_temps.reshape(self.geom.num_layers, self.geom.num_columns)

        #update observation well temps
        if do_update_obs_wells:
            self.update_obs_well_temps()

    def compute_jacobian_for_perm_powers(self, perm_powers_centre, delta=0.01, as_fraction=True, return_jac=False):
        '''
        self updates:
            via simulate:
                results -> results
                ss_temps -> ss_temps
                is_good_model -> is_good_model
            here:
                datfile permeabilities -> datfile permeabilities
        '''

        #do single model run for shape
        self.set_rock_permeabilities(perm_powers=perm_powers_centre)
        self.simulate(reshape=False, is_silent=True)
        T_centre = self.ss_temps

        try:
            r = re.compile("OBS*")
            obs_wells_list = filter(r.match, self.geom.well.keys())
        except AttributeError:
            print('No observation wells specified: adding manually')
            obs_wells_list = ['OBS 1', 'OBS 2', 'OBS 3',
                              'OBS 4', 'OBS 5', 'OBS 6', 'OBS 7']

        (d_0, T_0) = self.geom.well_values(obs_wells_list[0], T_centre, elevation=True)

        #assume same number observations per well
        jacobian = np.zeros((int(len(T_0) * len(obs_wells_list)), len(perm_powers_centre)))

        #loop over perturbations
        for j in range(0, len(perm_powers_centre)):

            #perturbations
            delta_vector = np.insert(np.zeros(len(perm_powers_centre) - 1), j, delta)
            perm_powers_j_back = (1 - delta_vector) * perm_powers_centre
            perm_powers_j_forward = (1 + delta_vector) * perm_powers_centre
            h_j = delta * perm_powers_centre[j]

            #run model at half back
            self.set_rock_permeabilities(perm_powers=perm_powers_j_back)
            self.simulate(reshape=False, is_silent=True)
            T_back = self.ss_temps

            #run model at half forwards
            self.set_rock_permeabilities(perm_powers=perm_powers_j_forward)
            self.simulate(reshape=False, is_silent=True)
            T_forward = self.ss_temps

            #get outputs and store differences
            # TODO: ignore first element...???
            T_diffs = np.zeros((len(obs_wells_list), len(T_0)))
            for i, welli in enumerate(obs_wells_list):
                (d_back_well, T_back_well) = self.geom.well_values(welli, T_back, elevation=True)
                (d_forward_well, T_forward_well) = self.geom.well_values(welli, T_forward, elevation=True)

                T_diffs[i, :] = T_forward_well - T_back_well

            jacobian[:, j] = T_diffs.flatten() / h_j

        self.jacobian = jacobian
        if return_jac:
            return jacobian


    def generate_synthetic_data(self,perm_powers_truths,do_plot=False,save_data=True):
        '''
        Generate synthetic well data. Note that this will be fine-scale, i.e. model scale. Will need to create a coarser version 
        using inverse core functionality to get proper data.
        To avoid having to re-simulate, save to a dict by default. Then reload when needed and pass to constructor.
        '''

        #time
        start = timeit.default_timer()

        if do_plot:
            self.geom.slice_plot('x', wells=True)
            plt.show()

        num_params = len(perm_powers_truths)
        print('perm powers (truths): ')
        print(perm_powers_truths)

        #run model for synthetic temps.
        self.set_rock_permeabilities(perm_powers=perm_powers_truths)
        self.simulate(reshape=False)

        #add noise
        print('using random seed')
        prng = RandomState(0)
        T_noise = prng.normal(0, 5., size=self.ss_temps.shape)
        self.ss_temps = self.ss_temps + T_noise
        self.update_obs_well_temps()
        #
        #self.T_noise = T_noise
        #self.T_measured = T_synthetic + T_noise
        #self.ss_temps = T_synthetic + T_noise #note add noise to ss temps

        #save synthetic data
        if save_data:
            save_path = './saved_data/'
            try:
                os.makedirs(save_path)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(save_path):
                    pass
                else:
                    raise
            synthetic_data = {}
            synthetic_data['T_measured'] = self.ss_temps
            synthetic_data['T_noise'] = T_noise
            # can't save geom object itself, just save name? Could save file?!
            synthetic_data['geom_measured_name'] = self.geom_name
            synthetic_data['T_obs_well'] = self.ss_temps_obs_well
            synthetic_data['d_obs_well'] = self.d_obs_well

            pickle.dump(synthetic_data, open(save_path + "synthetic_data" + ".p", "wb"))

            #load synthetic data with:
            #data_dict = pickle.load(open("./saved_data/synthetic_data.p", "rb" ))

        #time
        stop = timeit.default_timer()
        print('time to generate synthetic data (s): ')
        print(stop - start)

        return
