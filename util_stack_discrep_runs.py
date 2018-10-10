import numpy as np
import timeit
import pickle
import os
import errno
from lib import GeothermalCore as GC
from lib import InverseCore as IC
from scipy.optimize import minimize


#---create or load comparison space model (basis of likelihood function)
discrepancy_filenames = ['./data_dump/discrepancy_runs/discrepancies_00.p',
                         './data_dump/discrepancy_runs/discrepancies_01.p',
                         './data_dump/discrepancy_runs/discrepancies_02.p',
                         './data_dump/discrepancy_runs/discrepancies_03.p',
                         './data_dump/discrepancy_runs/discrepancies_04.p']



discrep_combined = pickle.load(open(discrepancy_filenames[0], "rb"))

for filename in discrepancy_filenames[1:]:
    discrep_next = pickle.load(open(filename, "rb"))
    discrep_combined = np.vstack((discrep_combined,discrep_next))


pickle.dump(discrep_combined, open('./saved_data/discrepancies_combined.p', "wb"))