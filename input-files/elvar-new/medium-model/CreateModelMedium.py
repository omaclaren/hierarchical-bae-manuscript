from t2data import *
from t2grids import *

print 'Generating Intermediate Grid With 340 blocks (20 atmospheric blocks)'

dx = [100.]*20
dy = [20.]
dz= [100.]*16
geo = mulgrid().rectangular(dx, dy, dz, atmos_type = 1, convention = 0)
print geo

# Generate well tracks:
Nwells = 7
for i in range(0,Nwells):
    w = well('OBS '+str(i+1), [[100.0 + i*300., 0.5, 0.0],[100.0 + i*300., 0.5, -1510.0]])
    geo.add_well(w)
#geo.translate([0.,0.,0.], wells=True)
geo.write('g2medium.dat')

dat = t2data()
dat.simulator = 'AUTOUGH2.2EW'
dat.grid = t2grid().fromgeo(geo)
dat.title = '2D medium model   1200kJ/kg inflow'

print "   *** Assign rocktypes to blocks  ***"
#ROCKS
rocklib = [['SURFA', .2500000E4, .100000001, 50.00E-15, 50.00E-15, 10.00E-15, .2500000E1, .1000000E4,], ['CAPRO', .2500000E4, .100000001,  1.00E-15,  1.00E-15,  0.50E-15, .2500000E1, .1000000E4], ['OUTFL', .2500000E4, .100000001,  1.00E-15,  1.00E-15, 20.00E-15, .2500000E1, .1000000E4], ['MEDM ', .2500000E4, .100000001, 25.00E-15, 25.00E-15, 10.00E-15, .2500000E1, .1000000E4], ['DEEP ', .2500000E4, .100000001,  0.50E-15,  0.50E-15,  0.50E-15, .2500000E1, .1000000E4], ['UPFLO', .2500000E4, .100000001,  0.50E-15,  0.50E-15, 10.00E-15, .2500000E1, .1000000E4]]
# Define rectangular areas for each rock type [x1, x2, z1, z2]
rockAreas = [[0., 2.e3, -2.e2, 0.], [0., 2.e3, -4.e2, -2.e2], [1.4e3, 1.6e3, -4.e2, -2.e2], [0., 2.e3, -1.e3, -4.e2], [0., 2.e3, -1.6e3, -1.e3], [0., 2.e2, -1.6e3, -1.e3]]
for i in range(0,6):
    rprops = rocklib[i]
    r = rocktype(name=rprops[0], permeability = [rprops[3], rprops[4], rprops[5]] )
    r.porosity = rprops[2]
    r.density = rprops[1]
    r.conductivity = rprops[6]
    r.specific_heat = rprops[7]
    dat.grid.add_rocktype(r)
    if i==0:
        # Start by assigning all blocks to rock-type SURFA
        for blk in dat.grid.blocklist[0:]:
            blk.rocktype = r
    else:
        for blk in dat.grid.blocklist[0:]:
            xyz = blk.centre
            RockRectangle = rockAreas[i]
            if (xyz[0] > RockRectangle[0])and(xyz[0] < RockRectangle[1])and(xyz[2] > RockRectangle[2])and(xyz[2] < RockRectangle[3]):
                blk.rocktype = r
dat.grid.clean_rocktypes()

print "   *** Setting up Simulation Parameters ***"
#Parameters
dat.parameter['max_iterations'] = 0
dat.parameter['print_level'] = 3
dat.parameter['max_timesteps'] = 500
dat.parameter['max_duration'] = 0
dat.parameter['print_interval'] = 500
dat.parameter['tstart'] = 0.0
dat.parameter['tstop'] = 1.E16
dat.parameter['const_timestep'] = 1.E8
dat.parameter['max_timestep'] = 1.E16

#dat.parameter['relative_error'] = 1.E-8
#dat.parameter['absolute_error'] = 1.0

dat.parameter['gravity'] = 9.81
dat.parameter['option'][1] = 1
dat.parameter['option'][11] = 2
dat.parameter['option'][12] = 2
dat.parameter['option'][16] = 5
dat.parameter['option'][23] = 1
dat.parameter['default_incons'] = [.10135E6, 15.]

#START
dat.start = True

print "   *** Setting up RPCAP Parameters  ***"
#RPCAP
dat.relative_permeability['type'] = 1.0
dat.relative_permeability['parameters'] = [0.5, 0.0, 1.0, 0.5, 0.0]
dat.capillarity['type'] = 1
dat.capillarity['parameters'] = [0.0, 0.0, 1.0, 0.0, 0.0]

print "   *** EOS Configuration  ***"
#MULTI
dat.multi['num_components'] = 1
dat.multi['num_equations'] = 2
dat.multi['num_phases'] = 2
dat.multi['num_secondary_parameters'] = 6
#dat.multi['eos'] = 'EWA'

print "   *** Linear Equation Solver Parameters  ***"
#LINEQ
dat.lineq['type'] = 2
dat.lineq['epsilon'] = 1e-11
dat.lineq['max_iterations'] = 999
dat.lineq['num_orthog'] = 5150
dat.lineq['gauss'] = 1


# Generators: add injection and production wells:
print "   *** Generators, i.e. bottom boundary conditions  ***"
dat.clear_generators()
# Add mass flux at the left end of bottom boundary (between 0m and 200m):
totalmassrate = 5.*0.03
enthalpy = 1200000.0
layer = geo.layerlist[-1] # bottom layer
cols = [col for col in geo.columnlist if 0. <= col.centre[0] <= 2.e2]
totalarea = sum([col.area for col in cols])
q = totalmassrate / totalarea
for col in cols:
    blockname = geo.block_name(layer.name, col.name)
    gen = t2generator(name = col.name+'97', block = blockname, type = 'MASS', gx = q*col.area, ex = enthalpy)
    dat.add_generator(gen)
# Add mass flux at the left end of bottom boundary (between 0m and 200m):
heatflux = 80.e-3 # 80 mW/m^2
layer = geo.layerlist[-1] # bottom layer
cols = [col for col in geo.columnlist if 2.e2 <= col.centre[0] <= 2.e3]
totalarea = sum([col.area for col in cols])
for col in cols:
    blockname = geo.block_name(layer.name, col.name)
    gen = t2generator(name = col.name+'98', block = blockname, type = 'HEAT', gx = heatflux*col.area)
    dat.add_generator(gen)


#dat.convert_to_AUTOUGH2(warn=True, MP=False, simulator='AUTOUGH2', eos='EW')
    
dat.write('2DM002.dat')













