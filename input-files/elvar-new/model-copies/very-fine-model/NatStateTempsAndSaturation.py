# plot permeabilites

import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import string
import h5py


font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

# set tick width
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.minor.size'] = 3
matplotlib.rcParams['xtick.minor.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.minor.size'] = 3
matplotlib.rcParams['ytick.minor.width'] = 1


nr, nc = 160, 200
adjfile = h5py.File('2DVF002.h5','r')
sat = adjfile['primary'][-1,nc::,1]
for i in range(0,len(sat)):
    if sat[i] > 1.0 :
        sat[i] = 0.0
sat = sp.reshape(sat,(nr,nc))

temp = adjfile['element'][-1,nc::,1]
#from t2incons import *
#inc = t2incon('savNat.save')
#temp = sp.zeros(32000)
#for i in range(200,len(temp)):
#    temp[i] = inc[i][1]


temp = sp.reshape(temp,(nr,nc))
sat = sp.reshape(sat,(nr,nc))



fig = plt.figure(1)
ax = plt.subplot(111)
cax = ax.matshow(temp,cmap='hot', interpolation='none', vmin=0.0, vmax=300.0, aspect=16./20.)

plt.xticks(sp.arange(0.0-0.5, 250.0-0.5, 50.0))
xlabels = [item.get_text() for item in ax.get_xticklabels()]
xlabels = sp.arange(0,2500,500)
ax.set_xticklabels(xlabels)
plt.yticks(sp.arange(0.0-0.5, 200.0-0.5, 40.0))
ylabels = [item.get_text() for item in ax.get_yticklabels()]
ylabels = -sp.arange(0,2000,400)
ax.set_yticklabels(ylabels)
ax.minorticks_on()

ax.xaxis.set_label_position('top') 
plt.xlabel('Horizontal Position [m]', labelpad=15)
plt.ylabel('Elevation [m]')

#plt.title('(a)', x=0.5, y=-0.25)
ax.text(-0.3, 1.15, '('+string.ascii_lowercase[0]+')', transform=ax.transAxes, size=25, weight='bold')
plt.subplots_adjust(hspace = .5, wspace = 0.5)


im = cax
#ax = plt.subplot(133)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.96, 0.18, 0.06, 0.63])
cbar = fig.colorbar(im,cax=cbar_ax, ticks=[0.0,50.0,100.0,150.0,200.0,250.0,300.0])
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

fig.savefig("NatStateTemp.png",bbox_inches='tight',dpi=200)
#plt.show()



fig = plt.figure(2)
ax = plt.subplot(111)
cax = ax.matshow(sat,cmap='Blues', interpolation='none', vmin=0.0, vmax=0.3, aspect=16./20.)

plt.xticks(sp.arange(0.0-0.5, 250.0-0.5, 50.0))
xlabels = [item.get_text() for item in ax.get_xticklabels()]
xlabels = sp.arange(0,2500,500)
ax.set_xticklabels(xlabels)
plt.yticks(sp.arange(0.0-0.5, 200.0-0.5, 40.0))
ylabels = [item.get_text() for item in ax.get_yticklabels()]
ylabels = -sp.arange(0,2000,400)
ax.set_yticklabels(ylabels)
ax.minorticks_on()

ax.xaxis.set_label_position('top') 
plt.xlabel('Horizontal Position [m]', labelpad=15)
plt.ylabel('Elevation [m]')

#plt.title('(b)', x=0.5, y=-0.25)
ax.text(-0.3, 1.15, '('+string.ascii_lowercase[1]+')', transform=ax.transAxes, size=25, weight='bold')
plt.subplots_adjust(hspace = .5, wspace = 0.5)

im = cax
#ax = plt.subplot(133)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.96, 0.18, 0.06, 0.63])
cbar = fig.colorbar(im,cax=cbar_ax, ticks=[0.0,0.1,0.2,0.3])
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

fig.savefig("NatStateSaturation.png",bbox_inches='tight',dpi=200)
#plt.show()




