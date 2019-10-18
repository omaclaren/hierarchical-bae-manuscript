import numpy as np
import timeit
import pickle
import os
import errno
from lib import GeothermalCore as GC
from lib import InverseCore as IC
from scipy.optimize import minimize


#---create or load comparison space model (basis of likelihood function)
load_discrepancy = True
#load_discrepancy = False

if load_discrepancy:
    #02 interrupted by power outage...
    flatchain_filenames = ['./data_dump/mcmc-with-discrep-project-data/sampler_flatchain_test_kerinci_bayes_model_discrep_00.p',
                           './data_dump/mcmc-with-discrep-project-data/sampler_flatchain_test_kerinci_bayes_model_discrep_01.p',
                           './data_dump/mcmc-with-discrep-project-data/sampler_flatchain_test_kerinci_bayes_model_discrep_03.p',
                           './data_dump/mcmc-with-discrep-project-data/sampler_flatchain_test_kerinci_bayes_model_discrep_04.p',
                           './data_dump/mcmc-with-discrep-project-data/sampler_flatchain_test_kerinci_bayes_model_discrep_05.p',
                           './data_dump/mcmc-with-discrep-project-data/sampler_flatchain_test_kerinci_bayes_model_discrep_06.p'
                           ]
else:
    #00 removed to be compatable with above.
    flatchain_filenames = ['./data_dump/mcmc_runs_naive_kerinci/sampler_flatchain_test_kerinci_bayes_model_01.p',
                            './data_dump/mcmc_runs_naive_kerinci/sampler_flatchain_test_kerinci_bayes_model_02.p',
                            './data_dump/mcmc_runs_naive_kerinci/sampler_flatchain_test_kerinci_bayes_model_03.p',
                            './data_dump/mcmc_runs_naive_kerinci/sampler_flatchain_test_kerinci_bayes_model_04.p',
                            './data_dump/mcmc_runs_naive_kerinci/sampler_flatchain_test_kerinci_bayes_model_05.p',
                            './data_dump/mcmc_runs_naive_kerinci/sampler_flatchain_test_kerinci_bayes_model_06.p']



flatchain_combined = pickle.load(open(flatchain_filenames[0], "rb"))

for filename in flatchain_filenames[1:]:
    flatchain_next = pickle.load(open(filename, "rb"))
    flatchain_combined = np.vstack((flatchain_combined,flatchain_next))


if load_discrepancy:
    pickle.dump(flatchain_combined, open(
        './saved_data/sampler_flatchain_test_kerinci_bayes_model_discrep_combined.p', "wb"))
else:
    pickle.dump(flatchain_combined, open(
        './saved_data/sampler_flatchain_test_kerinci_bayes_model_combined.p', "wb"))
