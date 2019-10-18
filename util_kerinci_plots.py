import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
from lib import InverseCore as IC
from lib import GeothermalCore as GC
from scipy import stats
import errno
import os
import pickle
import timeit
import numpy as np
import matplotlib as mpl
mpl.use("pgf")

#----
#save a default copy of plotting settings
rc_default = plt.rcParams.copy()

#----
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
#r"\setmathfont{xits-math.otf}",
#r"\setmainfont{'Computer Modern Roman'}", # serif font via preamble
#]
#plt.rcParams['image.cmap'] = 'Greys'

#----load models etc of interest

#HERE.


#----predictive checks


#----parameter plots
labels = {'title': '',
          'xlabel': r'Temperature ($^\circ$C)', r'ylabel': 'Elevation (m)'}
ticks = {'xticks_gap': 50, 'yticks_gap': 250}
