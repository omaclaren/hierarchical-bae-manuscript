# Run fwd simulation:
from t2data import *
import time

datfile = t2data('2DM002.dat')
temptime = time.clock()
datfile.run(save_filename='savNat', incon_filename='someInc', simulator='autough2_5_h5all.exe',silent=True) 
print 'time spent on fwd simulation', time.clock() - temptime