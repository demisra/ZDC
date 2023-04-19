from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import mm, cm, GeV, MeV, degree, rad
from math import pi
SIM = DD4hepSimulation()

energy = "100*GeV"
particle = "mu-"
count = "10e4"
format = "edm4hep"

ionCrossingAngle = -0.025*rad
ZDC_z_pos = 3549*cm
ZDC_x_pos = -88.74*cm
ZDC_y_pos = 0*cm

SIM.gun.position = (ZDC_x_pos, ZDC_y_pos, ZDC_z_pos)
SIM.gun.crossingAngleBoost = ionCrossingAngle
SIM.gun.thetaMin = ionCrossingAngle
SIM.gun.thetaMax = ionCrossingAngle
SIM.gun.phiMin = 0.
SIM.gun.phiMax = 0.
SIM.gun.distribution = "uniform"
SIM.gun.momentumMin = eval(energy)
SIM.gun.momentumMax = eval(energy)
SIM.gun.particle = particle
SIM.gun.multiplicity = 1

SIM.outputFile = f"zdc_{particle}_{energy.replace('*','')}_{count}.{format}.root"