# Run fwd simulation:
from t2data import *
import time

datfile = t2data('2DC002.dat')
temptime = time.clock()
datfile.run(save_filename='savNat', incon_filename='someInc', simulator='AUTOUGH2_6XD',silent=True)
print 'time spent on fwd simulation', time.clock() - temptime
