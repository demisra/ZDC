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
from sklearn.neighbors import KernelDensity

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


# %%
def yIDDist_WSi(data, branch, events):
    yID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        event_yID = signedint(bitExtract(event_cellID, 12, 36))
        for i in range(len(event_cellID)):
            if event_layerID[i] in [1, 12, 23]:
                event_yID[i] = 0.3 * event_yID[i]
        yID.extend(event_yID)
    return np.array(yID)


# %%
def xIDDist_PbSi(data, branch, events):
    xID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_xID = signedint(bitExtract(event_cellID, 12, 24))
        xID.extend(event_xID)
    return np.array(xID)


# %%
def yIDDist_PbSi(data, branch, events):
    yID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_yID = signedint(bitExtract(event_cellID, 12, 36))
        yID.extend(event_yID)
    return np.array(yID)


# %%
def xIDDist_PbScint(data, branch, events):
    xID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_xID = 10 * signedint(bitExtract(event_cellID, 12, 24))
        xID.extend(event_xID)
    return np.array(xID)


# %%
def yIDDist_PbScint(data, branch, events):
    yID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_yID = 10 * signedint(bitExtract(event_cellID, 12, 36))
        yID.extend(event_yID)
    return np.array(yID)


# %%
WSi_xID_10GeV = xIDDist_WSi(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_layerID_10GeV = layerIDDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_hitsEnergy_10GeV = 101*energyDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)

WSi_xID_20GeV = xIDDist_WSi(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_layerID_20GeV = layerIDDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_hitsEnergy_20GeV = 101*energyDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)

WSi_xID_50GeV = xIDDist_WSi(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_layerID_50GeV = layerIDDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_hitsEnergy_50GeV = 101*energyDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)

WSi_xID_100GeV = xIDDist_WSi(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_layerID_100GeV = layerIDDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_hitsEnergy_100GeV = 101*energyDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)

WSi_xID_150GeV = xIDDist_WSi(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_layerID_150GeV = layerIDDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
WSi_hitsEnergy_150GeV = 101*energyDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)

# %%
PbSi_xID_10GeV = xIDDist_PbSi(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)
PbSi_layerID_10GeV = layerIDDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000) + 23
PbSi_hitsEnergy_10GeV = 403*energyDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)

PbSi_xID_20GeV = xIDDist_PbSi(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)
PbSi_layerID_20GeV = layerIDDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000) + 23
PbSi_hitsEnergy_20GeV = 403*energyDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)

PbSi_xID_50GeV = xIDDist_PbSi(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)
PbSi_layerID_50GeV = layerIDDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000) + 23
PbSi_hitsEnergy_50GeV = 403*energyDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)

PbSi_xID_100GeV = xIDDist_PbSi(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)
PbSi_layerID_100GeV = layerIDDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000) + 23
PbSi_hitsEnergy_100GeV = 403*energyDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)

PbSi_xID_150GeV = xIDDist_PbSi(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)
PbSi_layerID_150GeV = layerIDDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000) + 23
PbSi_hitsEnergy_150GeV = 403*energyDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)

# %%
PbScint_xID_10GeV = xIDDist_PbScint(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000)
PbScint_layerID_10GeV = layerIDDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000) + 35
PbScint_hitsEnergy_10GeV = 60*energyDist(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000)

PbScint_xID_20GeV = xIDDist_PbScint(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000)
PbScint_layerID_20GeV = layerIDDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000) + 35
PbScint_hitsEnergy_20GeV = 60*energyDist(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000)

PbScint_xID_50GeV = xIDDist_PbScint(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000)
PbScint_layerID_50GeV = layerIDDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000) + 35
PbScint_hitsEnergy_50GeV = 60*energyDist(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000)

PbScint_xID_100GeV = xIDDist_PbScint(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000)
PbScint_layerID_100GeV = layerIDDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000) + 35
PbScint_hitsEnergy_100GeV = 60*energyDist(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000)

PbScint_xID_150GeV = xIDDist_PbScint(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000)
PbScint_layerID_150GeV = layerIDDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000) + 35
PbScint_hitsEnergy_150GeV = 60*energyDist(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDCHcalHits", 10000)

# %%
xID_10GeV = np.concatenate((WSi_xID_10GeV, PbSi_xID_10GeV, PbScint_xID_10GeV))
layerID_10GeV = np.concatenate((WSi_layerID_10GeV, PbSi_layerID_10GeV, PbScint_layerID_10GeV))
hitsEnergy_10GeV = np.concatenate((WSi_hitsEnergy_10GeV, PbSi_hitsEnergy_10GeV, PbScint_hitsEnergy_10GeV))

xID_20GeV = np.concatenate((WSi_xID_20GeV, PbSi_xID_20GeV, PbScint_xID_20GeV))
layerID_20GeV = np.concatenate((WSi_layerID_20GeV, PbSi_layerID_20GeV, PbScint_layerID_20GeV))
hitsEnergy_20GeV = np.concatenate((WSi_hitsEnergy_20GeV, PbSi_hitsEnergy_20GeV, PbScint_hitsEnergy_20GeV))

xID_50GeV = np.concatenate((WSi_xID_50GeV, PbSi_xID_50GeV, PbScint_xID_50GeV))
layerID_50GeV = np.concatenate((WSi_layerID_50GeV, PbSi_layerID_50GeV, PbScint_layerID_50GeV))
hitsEnergy_50GeV = np.concatenate((WSi_hitsEnergy_50GeV, PbSi_hitsEnergy_50GeV, PbScint_hitsEnergy_50GeV))

xID_100GeV = np.concatenate((WSi_xID_100GeV, PbSi_xID_100GeV, PbScint_xID_100GeV))
layerID_100GeV = np.concatenate((WSi_layerID_100GeV, PbSi_layerID_100GeV, PbScint_layerID_100GeV))
hitsEnergy_100GeV = np.concatenate((WSi_hitsEnergy_100GeV, PbSi_hitsEnergy_100GeV, PbScint_hitsEnergy_100GeV))

xID_150GeV = np.concatenate((WSi_xID_150GeV, PbSi_xID_150GeV, PbScint_xID_150GeV))
layerID_150GeV = np.concatenate((WSi_layerID_150GeV, PbSi_layerID_150GeV, PbScint_layerID_150GeV))
hitsEnergy_150GeV = np.concatenate((WSi_hitsEnergy_150GeV, PbSi_hitsEnergy_150GeV, PbScint_hitsEnergy_150GeV))

# %%
plt.hist(layerID_10GeV, 65, histtype='step');
plt.hist(layerID_20GeV, 65, histtype='step');
plt.hist(layerID_50GeV, 65, histtype='step');
plt.hist(layerID_100GeV, 65, histtype='step');
plt.hist(layerID_150GeV, 65, histtype='step');
plt.xlabel('layerID')
plt.ylabel('Hits')
plt.title("LayerID Hit Distribution")

# %%
df = pd.DataFrame([xID_10GeV, layerID_10GeV, hitsEnergy_10GeV]).T

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
plt.contour(Yi, Xi, Z, alpha=0.7, cmap=plt.cm.jet, levels=20);
plt.colorbar()

# %%
plt.scatter(dfreset['x'], dfreset['y'], marker='s', s=10, c=dfreset['z'], cmap=plt.cm.jet, norm=colors.LogNorm())
plt.colorbar()
plt.xlabel('x (cm)')
plt.ylabel('layerID')
plt.title('Calibrated Energy Deposition in the X-Z Plane')

# %%
plt.scatter(dfreset['x'], dfreset['y'], marker='s', s=10, c=dfreset['z'], cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel('x (cm)')
plt.ylabel('layerID')
plt.title('Calibrated Energy Deposition in the X-Z Plane')

# %% [markdown]
# Contour Plots

# %%
calibration_dict = {'ZDC_WSi_Hits':101.0675, 'ZDC_PbSi_Hits':402.7879, 'ZDCHcalHits':60.0160}


# %%
def energyContour(energy, branch, count):
    xID = xIDDist_WSi(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
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
    xID = xIDDist_WSi(samples[f'zdc_neutron_{energy}GeV_10e4.edm4hep.root'], f"{branch}", count)
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


# %%
energyColor(100, "ZDC_PbSi_Hits", 10000)

# %%
