# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import uproot as ur
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from bitstring import BitArray

# %%
paths = []

for (path, dirnames, filenames) in os.walk('/home/dmisra/eic/zdc_neutron_samples/'):
    paths.extend(os.path.join(path, name) for name in filenames)

# %%
samples = {}

for path in paths:
    with ur.open(path) as file:
       tree = file["events"]
       samples[os.path.basename(f'{path}')] = tree.arrays()

# %%
CrossingAngle = -0.025

def bitExtract(n, k, p):  
    return (((1 << k) - 1)  &  (n >> p))

#Get RMS90 of distribution
def is_outlier(df):
    p_05 = df.quantile(.05)
    p_95 = df.quantile(.95)
    return ~df.between(p_05, p_95)

#Extract signed integer from bitstring
def signedint(xbits):
    x_int = []
    x_bin = np.vectorize(np.binary_repr, otypes=[str])(xbits, width=12)
    for bits in x_bin:
            x_int.append(BitArray(bin=bits).int)
    return np.array(x_int)


# %%
branches = ['ZDC_SiliconPix_Hits','ZDCEcalHits','ZDC_WSi_Hits', 'ZDC_PbSi_Hits', 'ZDCHcalHits']


# %% [markdown]
# Functions

# %%
def countHits(data, branch, events):
    nhits = []
    for i in range(events):
        nhits.append(len(data[f"{branch}.energy"][i]))

    return nhits


# %%
def maxEnergy(data, branch, events):
    maximums = []
    for i in range(events):
        maximums.append(max(data[f"{branch}.energy"][i], default=0))

    return np.array(list(filter(lambda x: x != 0., maximums)))


# %%
def minEnergy(data, branch, events):
    minimums = []
    for i in range(events):
        minimums.append(min(data[f"{branch}.energy"][i], default=0))

    return np.array(minimums)


# %%
def layerIDDist(data, branch, events):
    layerID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        layerID.extend(event_layerID)
    return np.array(layerID)


# %%
def energyDist(data, branch, events):
    hitsEnergy = []
    for i in range(events):
        event_hitsEnergy = np.array(data[f"{branch}.energy"][i])
        hitsEnergy.extend(event_hitsEnergy)
    return np.array(hitsEnergy)


# %%
def xIDDist(data, branch, events):
    xID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_xID = signedint(bitExtract(event_cellID, 12, 24))
        xID.extend(event_xID)
    return np.array(xID)


# %%
def xIDDist_WSi(data, branch, events):
    xID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        event_xID = signedint(bitExtract(event_cellID, 12, 24))
        for i in range(len(event_cellID)):
            if event_layerID[i] in [1, 12, 23]:
                event_xID[i] = 0.3 * event_xID[i]
        xID.extend(event_xID)
    return np.array(xID)


# %% [markdown]
# xID Distributions

# %%
plt.hist(xIDDist_WSi(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), 61, histtype="step", align="mid")
plt.hist(xIDDist_WSi(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), 61, histtype="step", align="mid")
plt.hist(xIDDist_WSi(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), 61, histtype="step", align="mid")
plt.hist(xIDDist_WSi(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), 61, histtype="step", align="mid")
plt.hist(xIDDist_WSi(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), 61, histtype="step", align="mid")
plt.xlabel("xID")
plt.ylabel("Events")
plt.title("W/Si xID Distribution")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.semilogy()

# %%
plt.hist(xIDDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), 61, histtype="step", align="mid")
plt.hist(xIDDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), 61, histtype="step", align="mid")
plt.hist(xIDDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), 61, histtype="step", align="mid")
plt.hist(xIDDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), 61, histtype="step", align="mid")
plt.hist(xIDDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), 61, histtype="step", align="mid")
plt.xlabel("xID")
plt.ylabel("Events")
plt.title("Pb/Si xID Distribution")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.semilogy()

# %%
plt.hist(xIDDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), 7, histtype="step", align="mid")
plt.hist(xIDDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), 7, histtype="step", align="mid")
plt.hist(xIDDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), 7, histtype="step", align="mid")
plt.hist(xIDDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), 7, histtype="step", align="mid")
plt.hist(xIDDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), 7, histtype="step", align="mid")
plt.xlabel("xID")
plt.ylabel("Events")
plt.title("Pb/Scint xID Distribution")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.semilogy()

# %% [markdown]
# layerID Distributions

# %%
std_WSi_10GeV = np.std(layerIDDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000))
std_WSi_20GeV = np.std(layerIDDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000))
std_WSi_50GeV = np.std(layerIDDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000))
std_wSi_100GeV = np.std(layerIDDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000))
std_WSi_150GeV = np.std(layerIDDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000))

# %%
plt.hist(layerIDDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), 23, histtype="step")
plt.hist(layerIDDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), 23, histtype="step")
plt.hist(layerIDDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), 23, histtype="step")
plt.hist(layerIDDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), 23, histtype="step")
plt.hist(layerIDDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), 23, histtype="step")
plt.xlabel("LayerID")
plt.ylabel("Events")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.title("W/Si Hit LayerID Distribution")

# %%
plt.hist(layerIDDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), 12, histtype="step", align="mid")
plt.hist(layerIDDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), 12, histtype="step", align="mid")
plt.hist(layerIDDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), 12, histtype="step", align="mid")
plt.hist(layerIDDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), 12, histtype="step", align="mid")
plt.hist(layerIDDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), 12, histtype="step", align="mid")
plt.xlabel("LayerID")
plt.ylabel("Events")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.title("Pb/Si Hit LayerID Distribution")

# %%
plt.hist(layerIDDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), 30, histtype="step", align="mid")
plt.hist(layerIDDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), 30, histtype="step", align="mid")
plt.hist(layerIDDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), 30, histtype="step", align="mid")
plt.hist(layerIDDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), 30, histtype="step", align="mid")
plt.hist(layerIDDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), 30, histtype="step", align="mid")
plt.xlabel("LayerID")
plt.ylabel("Events")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.title("Pb/Scint Hit LayerID Distribution")

# %% [markdown]
# Dynamic Range

# %%
plt.hist(maxEnergy(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDCEcalHits", 10000), bins=np.linspace(0,25, num=25), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDCEcalHits", 10000), bins=np.linspace(0,25, num=25), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDCEcalHits", 10000), bins=np.linspace(0,25, num=25), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDCEcalHits", 10000), bins=np.linspace(0,25, num=25), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDCEcalHits", 10000), bins=np.linspace(0,25, num=25), histtype="step")
plt.xlabel("Maximum Energy Deposition")
plt.ylabel("Events")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.semilogy()
plt.title("PbWO4 Crystal Maximum Energy Distribution")

# %%
plt.hist(maxEnergy(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), bins=np.linspace(0,0.15, num=15), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), bins=np.linspace(0,0.15, num=15), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), bins=np.linspace(0,0.15, num=15), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), bins=np.linspace(0,0.15, num=15), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000), bins=np.linspace(0,0.15, num=15), histtype="step")
plt.xlabel("Maximum Energy Deposition")
plt.ylabel("Events")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.semilogy()
plt.title("W/Si Maximum Energy Distribution")

# %%
plt.hist(maxEnergy(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), bins=np.linspace(0,0.15, num=15), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), bins=np.linspace(0,0.15, num=15), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), bins=np.linspace(0,0.15, num=15), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), bins=np.linspace(0,0.15, num=15), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000), bins=np.linspace(0,0.15, num=15), histtype="step")
plt.xlabel("Maximum Energy Deposition")
plt.ylabel("Events")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.semilogy()
plt.title("Pb/Si Maximum Energy Distribution")

# %%
plt.hist(maxEnergy(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), bins=np.linspace(0,0.5, num=20), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), bins=np.linspace(0,0.5, num=20), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), bins=np.linspace(0,0.5, num=20), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), bins=np.linspace(0,0.5, num=20), histtype="step")
plt.hist(maxEnergy(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000), bins=np.linspace(0,0.5, num=20), histtype="step")
plt.xlabel("Maximum Energy Deposition")
plt.ylabel("Events")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.semilogy()
plt.title("Pb/Scint Maximum Energy Distribution")

# %% [markdown]
# Hit Count Distributions

# %%
plt.hist(countHits(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], 'ZDC_WSi_Hits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], 'ZDC_WSi_Hits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], 'ZDC_WSi_Hits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], 'ZDC_WSi_Hits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], 'ZDC_WSi_Hits', 10000), 30, histtype = "step")
plt.xlabel("Hits in W/Si")
plt.ylabel("Events")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])

# %%
plt.hist(countHits(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], 'ZDC_PbSi_Hits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], 'ZDC_PbSi_Hits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], 'ZDC_PbSi_Hits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], 'ZDC_PbSi_Hits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], 'ZDC_PbSi_Hits', 10000), 30, histtype = "step")
plt.xlabel("Hits in Pb/Si")
plt.ylabel("Events")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])

# %%
plt.hist(countHits(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], 'ZDCHcalHits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], 'ZDCHcalHits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], 'ZDCHcalHits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], 'ZDCHcalHits', 10000), 30, histtype = "step")
plt.hist(countHits(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], 'ZDCHcalHits', 10000), 30, histtype = "step")
plt.xlabel("Hits in Pb/Scint")
plt.ylabel("Events")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])

# %% [markdown]
# Contour Plot Testing

# %%
from sklearn.neighbors import KernelDensity

# %%
WSi_xID = xIDDist_WSi(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_layerID = layerIDDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_hitsEnergy = energyDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)

# %%
df = pd.DataFrame([WSi_xID, WSi_layerID, WSi_hitsEnergy]).T

# %%
df.columns = ["x", "y", "z"]
df = df.sort_values(by=["x", "y"])

# %%
dfreset = df.groupby(["x", "y"]).mean().reset_index()
dfpivot=dfreset.pivot('x', 'y')

# %%
X=dfpivot.columns.levels[1].values
Y=dfpivot.index.values
Z=dfpivot.values
Xi,Yi = np.meshgrid(X, Y)

# %%
plt.scatter(dfreset['x'], dfreset['y'], s=10, c=dfreset['z'], cmap=plt.cm.jet, norm=colors.LogNorm())

# %%
xID = xIDDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
layerID = layerIDDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
energyDep = energyDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)

energyContained = 0.
totalEnergy = sum(energyDep)
while energyContained < 0.1*totalEnergy:
    
    

# %% [markdown]
# Contour Plots

# %%
cs = plt.contour(Yi, Xi, Z, alpha=0.7, cmap=plt.cm.jet, levels=10);

for i in range(10):
    contour = cs.collections[i]
    x=contour.vertices[:,0]
    y=contour.vertices[:,1]
    area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
    area=np.abs(area)

# %%
calibration_dict = {'ZDC_WSi_Hits':101.0675, 'ZDC_PbSi_Hits':402.7879, 'ZDCHcalHits':60.0160}


# %%
def energyContour(energy, branch, count):
    xID = xIDDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
    layerID = layerIDDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
    energyDep = energyDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)

    df = pd.DataFrame([xID, layerID, energyDep]).T
    df.columns = ["x", "y", "z"]
    df = df.sort_values(by=["x", "y"])
    dfreset = df.groupby(["x", "y"]).sum().reset_index()
    dfpivot=dfreset.pivot('x', 'y')

    X=dfpivot.columns.levels[1].values
    Y=dfpivot.index.values
    Z=dfpivot.values
    
    Xi,Yi = np.meshgrid(X, Y)
    plt.contour(Yi, Xi, Z, alpha=0.7, cmap=plt.cm.jet, levels=20);
    plt.colorbar()


# %%
def hitContour(energy, branch, count):
    xID = xIDDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
    layerID = layerIDDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
    hitsCount = countHits(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)

    df = pd.DataFrame([xID, layerID, hitsCount]).T
    df.columns = ["x", "y", "z"]
    df = df.sort_values(by=["x", "y"])
    dfreset = df.groupby(["x", "y"]).sum().reset_index()
    dfpivot=dfreset.pivot('x', 'y')

    X=dfpivot.columns.levels[1].values
    Y=dfpivot.index.values
    Z=dfpivot.values
    
    Xi,Yi = np.meshgrid(X, Y)
    plt.contour(Yi, Xi, Z, alpha=0.7, cmap=plt.cm.jet, levels=20);
    plt.colorbar()


# %%
energyContour(100, "ZDC_PbSi_Hits", 10000)

# %%
energyContour(100, "ZDCHcalHits", 10000)

# %%
hitContour(100, "ZDC_PbSi_Hits", 10000)

# %%
hitContour(100, "ZDCHcalHits", 10000)


# %%
def energyColor(energy, branch, count):
    xID = xIDDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
    layerID = layerIDDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
    energyDep = energyDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
    
    xID = xIDDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
    layerID = layerIDDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
    energyDep = energyDist(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)

    df = pd.DataFrame([xID, layerID, energyDep]).T
    df.columns = ["x", "y", "z"]
    df = df.sort_values(by=["x", "y"])
    dfreset = df.groupby(["x", "y"]).sum().reset_index()
    dfpivot=dfreset.pivot('x', 'y')

    X=dfpivot.columns.levels[1].values
    Y=dfpivot.index.values
    Z=dfpivot.values

    plt.scatter(X, Y, c=Z, cmap = plt.cm.jet)
