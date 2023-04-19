from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import mm, cm, GeV, MeV, degree, rad
SIM = DD4hepSimulation()

energy = "100*GeV"
particle = "neutron"
count = "10e4"
format = "edm4hep"

ionCrossingAngle = -0.025*rad

SIM.gun.position = (0.,0.,0.)
SIM.gun.crossingAngleBoost = ionCrossingAngle
SIM.gun.thetaMin = ionCrossingAngle + 0.002
SIM.gun.thetaMax = ionCrossingAngle + 0.002
SIM.gun.phiMin = 0.
SIM.gun.phiMax = 0.
SIM.gun.distribution = "uniform"
SIM.gun.momentumMin = eval(energy)
SIM.gun.momentumMax = eval(energy)
SIM.gun.particle = particle
SIM.gun.multiplicity = 1

SIM.outputFile = f"zdc_{particle}_{energy.replace('*','')}_{count}.{format}.root"