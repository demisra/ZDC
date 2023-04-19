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
#     display_name: pygenv
#     language: python
#     name: python3
# ---

# %%
import os
import uproot as ur
import numpy as np
from bitstring import BitArray
import pandas as pd

# %% [markdown]
# Load Data from ROOT Files

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

# %% [markdown]
# Detector Parameters

# %%
pixel_layer_positions = [1, 12, 23]


# %% [markdown]
# Helper Functions

# %%
def bitExtract(n, k, p):  
    return (((1 << k) - 1)  &  (n >> p))

#Extract signed integer from bitstring
def signedint(xbits):
    x_int = []
    x_bin = np.vectorize(np.binary_repr, otypes=[str])(xbits, width=12)
    for bits in x_bin:
            x_int.append(BitArray(bin=bits).int)
    return np.array(x_int)


# %%
def crystalPos(crystalID: int):
    x = ((crystalID - 1) % 20) - 9.5
    y = np.floor((crystalID - 1) / 20) - 9.5
    y = -y
    return x, y


# %%
def get_labels(data, count):
    energy_labels = []
    for i in range(count):
        label = np.sqrt(data["MCParticles.momentum.x"][0,0]**2 + data["MCParticles.momentum.y"][0,0]**2 + data["MCParticles.momentum.z"][0,0]**2)
        
        if len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            energy_labels.append(label)
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            energy_labels.append(label)
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            energy_labels.append(label)
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            energy_labels.append(label)
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            energy_labels.append(label)

    return energy_labels


# %%
def get_layerIDs(data, branch, events):
    layerID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)

        if len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            layerID.append(event_layerID)
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            layerID.append(event_layerID)
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            layerID.append(event_layerID)
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            layerID.append(event_layerID)
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            layerID.append(event_layerID)
            
    return layerID


# %%
def get_eDep(data, branch, events):
    hitsEnergy = []
    for i in range(events):
        event_hitsEnergy = np.array(data[f"{branch}.energy"][i])

        if len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            hitsEnergy.append(event_hitsEnergy)
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            hitsEnergy.append(event_hitsEnergy)
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            hitsEnergy.append(event_hitsEnergy)
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            hitsEnergy.append(event_hitsEnergy)
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            hitsEnergy.append(event_hitsEnergy)

    return hitsEnergy


# %%
def get_xIDs(data, branch, events):
    xID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_xID = signedint(bitExtract(event_cellID, 12, 24))

        if len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            xID.append(event_xID)
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            xID.append(event_xID)
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            xID.append(event_xID)
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            xID.append(event_xID)
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            xID.append(event_xID)
            
    return xID


# %%
def get_xIDs_WSi(data, branch, events):
    xID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        event_xID = signedint(bitExtract(event_cellID, 12, 24))
        for i in range(len(event_layerID)):
            if event_layerID[i] in pixel_layer_positions:
                event_xID[i] = 0.3 * event_xID[i]

        if len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            xID.append(event_xID)
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            xID.append(event_xID)
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            xID.append(event_xID)
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            xID.append(event_xID)
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            xID.append(event_xID)
            
    return xID


# %%
def get_yIDs(data, branch, events):
    yID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_yID = signedint(bitExtract(event_cellID, 12, 36))

        if len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            yID.append(event_yID)
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            yID.append(event_yID)
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            yID.append(event_yID)
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            yID.append(event_yID)
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            yID.append(event_yID)
            
    return yID


# %%
def get_yIDs_WSi(data, branch, events):
    yID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        event_yID = signedint(bitExtract(event_cellID, 12, 36))
        for i in range(len(event_layerID)):
            if event_layerID[i] in pixel_layer_positions:
                event_yID[i] = 0.3 * event_yID[i]
        
        if len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            yID.append(event_yID)
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            yID.append(event_yID)
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            yID.append(event_yID)
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) >0: 
            yID.append(event_yID)
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            yID.append(event_yID)

    return yID


# %%
def PbScint_features(data, branch, events): 
    xID = []
    yID = []
    layerID = []
    eDep = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        event_xID = 20*signedint(bitExtract(event_cellID, 12, 24))
        event_yID = 20*signedint(bitExtract(event_cellID, 12, 36))
        event_hitsEnergy = np.array(data[f"{branch}.energy"][i])
        df = pd.DataFrame({'x': event_xID, 'y': event_yID, 'layer': event_layerID, 'energy': event_hitsEnergy})
        df_grouped = df.groupby(['x','y', 'layer']).sum().reset_index()
        xIDs_integrated = df_grouped['x'].values
        yIDs_integrated = df_grouped['y'].values
        layerIDs_integrated = df_grouped['layer'].values
        eDep_integrated = df_grouped['energy'].values

        if len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            layerID.append(layerIDs_integrated)
            eDep.append(eDep_integrated)
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            layerID.append(layerIDs_integrated)
            eDep.append(eDep_integrated)
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            layerID.append(layerIDs_integrated)
            eDep.append(eDep_integrated)
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            layerID.append(layerIDs_integrated)
            eDep.append(eDep_integrated)
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            layerID.append(layerIDs_integrated)
            eDep.append(eDep_integrated)

    return xID, yID, layerID, eDep



# %%
def crystal_features(data, branch, events): 
    xID = []
    yID = []
    eDep = []
    layerID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_crystalID = signedint(bitExtract(event_cellID, 12, 10))
        event_xID, event_yID = crystalPos(event_crystalID)
        event_hitsEnergy = np.array(data[f"{branch}.energy"][i])
        df = pd.DataFrame({'x': event_xID, 'y': event_yID, 'energy': event_hitsEnergy})
        df_grouped = df.groupby(['x','y']).sum().reset_index()
        xIDs_integrated = df_grouped['x'].values
        yIDs_integrated = df_grouped['y'].values
        eDep_integrated = df_grouped['energy'].values

        if len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            eDep.append(eDep_integrated)
            layerID.append(np.zeros(len(eDep_integrated)))
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            eDep.append(eDep_integrated)
            layerID.append(np.zeros(len(eDep_integrated)))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            eDep.append(eDep_integrated)
            layerID.append(7*np.random.rand(len(eDep_integrated)))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            eDep.append(eDep_integrated)
            layerID.append(np.zeros(len(eDep_integrated)))
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            eDep.append(eDep_integrated)
            layerID.append(np.zeros(len(eDep_integrated)))

    return xID, yID, layerID, eDep


# %% [markdown]
# Dataset

# %%
import pickle

# %%
nevents = 10000

# %%
hitEnergyDep = dict()
xIDs = dict()
yIDs = dict()
layerIDs = dict()

# %%
crystal_features_10 = crystal_features(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDCEcalHits", nevents)
crystal_features_20 = crystal_features(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDCEcalHits", nevents)
crystal_features_50 = crystal_features(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDCEcalHits", nevents)
crystal_features_100 = crystal_features(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDCEcalHits", nevents)
crystal_features_150 = crystal_features(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDCEcalHits", nevents)

# %%
PbScint_features_10 = PbScint_features(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDCHcalHits", nevents)
PbScint_features_20 = PbScint_features(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDCHcalHits", nevents)
PbScint_features_50 = PbScint_features(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDCHcalHits", nevents)
PbScint_features_100 = PbScint_features(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDCHcalHits", nevents)
PbScint_features_150 = PbScint_features(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDCHcalHits", nevents)

# %%
eDep_SiPix_10 = get_eDep(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
eDep_WSi_10 = get_eDep(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
eDep_PbSi_10 = get_eDep(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
eDep_PbScint_10 = PbScint_features_10[3]
eDep_crystal_10 = crystal_features_10[3]

xIDs_SiPix_10 = get_xIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
xIDs_WSi_10 = get_xIDs_WSi(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
xIDs_PbSi_10 = get_xIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
xIDs_PbScint_10 = PbScint_features_10[0]
xIDs_crystal_10 = crystal_features_10[0]

yIDs_SiPix_10 = get_yIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
yIDs_WSi_10 = get_yIDs_WSi(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
yIDs_PbSi_10 = get_yIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
yIDs_PbScint_10 = PbScint_features_10[1]
yIDs_crystal_10 = crystal_features_10[1]

layerIDs_SiPix_10 = get_layerIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
layerIDs_crystal_10 = [x + 1 for x in crystal_features_10[2]]
layerIDs_WSi_10 = [x + 2 for x in get_layerIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)]
layerIDs_PbSi_10 = [x + 25 for x in get_layerIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]
layerIDs_PbScint_10 = [x + 37 for x in PbScint_features_10[2]]

# %%
hitEnergyDep['10GeV'] = [np.concatenate([eDep_SiPix_10[i], eDep_WSi_10[i], eDep_PbSi_10[i], eDep_PbScint_10[i], eDep_crystal_10[i]]) for i in range(len(eDep_WSi_10))]
xIDs['10GeV'] = [np.concatenate([xIDs_SiPix_10[i], xIDs_WSi_10[i], xIDs_PbSi_10[i], xIDs_PbScint_10[i], xIDs_crystal_10[i]]) for i in range(len(eDep_WSi_10))]
yIDs['10GeV'] = [np.concatenate([yIDs_SiPix_10[i], yIDs_WSi_10[i], yIDs_PbSi_10[i], yIDs_PbScint_10[i], yIDs_crystal_10[i]]) for i in range(len(eDep_WSi_10))]
layerIDs['10GeV'] = [np.concatenate([layerIDs_SiPix_10[i], layerIDs_WSi_10[i], layerIDs_PbSi_10[i], layerIDs_PbScint_10[i], layerIDs_crystal_10[i]]) for i in range(len(eDep_WSi_10))]

# %%
eDep_SiPix_20 = get_eDep(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
eDep_WSi_20 = get_eDep(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
eDep_PbSi_20 = get_eDep(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
eDep_PbScint_20 = PbScint_features_20[3]
eDep_crystal_20 = crystal_features_20[3]

xIDs_SiPix_20 = get_xIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
xIDs_WSi_20 = get_xIDs_WSi(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
xIDs_PbSi_20 = get_xIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
xIDs_PbScint_20 = PbScint_features_20[0]
xIDs_crystal_20 = crystal_features_20[0]

yIDs_SiPix_20 = get_yIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
yIDs_WSi_20 = get_yIDs_WSi(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
yIDs_PbSi_20 = get_yIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
yIDs_PbScint_20 = PbScint_features_20[1]
yIDs_crystal_20 = crystal_features_20[1]

layerIDs_SiPix_20 = get_layerIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
layerIDs_crystal_20 = [x + 1 for x in crystal_features_20[2]]
layerIDs_WSi_20 = [x + 2 for x in get_layerIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)]
layerIDs_PbSi_20 = [x + 25 for x in get_layerIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]
layerIDs_PbScint_20 = [x + 37 for x in PbScint_features_20[2]]

# %%
hitEnergyDep['20GeV'] = [np.concatenate([eDep_SiPix_20[i], eDep_WSi_20[i], eDep_PbSi_20[i], eDep_PbScint_20[i], eDep_crystal_20[i]]) for i in range(len(eDep_WSi_20))]
xIDs['20GeV'] = [np.concatenate([xIDs_SiPix_20[i], xIDs_WSi_20[i], xIDs_PbSi_20[i], xIDs_PbScint_20[i], xIDs_crystal_20[i]]) for i in range(len(eDep_WSi_20))]
yIDs['20GeV'] = [np.concatenate([yIDs_SiPix_20[i], yIDs_WSi_20[i], yIDs_PbSi_20[i], yIDs_PbScint_20[i], yIDs_crystal_20[i]]) for i in range(len(eDep_WSi_20))]
layerIDs['20GeV'] = [np.concatenate([layerIDs_SiPix_20[i], layerIDs_WSi_20[i], layerIDs_PbSi_20[i], layerIDs_PbScint_20[i], layerIDs_crystal_20[i]]) for i in range(len(eDep_WSi_20))]

# %%
eDep_SiPix_50 = get_eDep(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
eDep_WSi_50 = get_eDep(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
eDep_PbSi_50 = get_eDep(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
eDep_PbScint_50 = PbScint_features_50[3]
eDep_crystal_50 = crystal_features_50[3]

xIDs_SiPix_50 = get_xIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
xIDs_WSi_50 = get_xIDs_WSi(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
xIDs_PbSi_50 = get_xIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
xIDs_PbScint_50 = PbScint_features_50[0]
xIDs_crystal_50 = crystal_features_50[0]

yIDs_SiPix_50 = get_yIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
yIDs_WSi_50 = get_yIDs_WSi(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
yIDs_PbSi_50 = get_yIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
yIDs_PbScint_50 = PbScint_features_50[1]
yIDs_crystal_50 = crystal_features_50[1]

layerIDs_SiPix_50 = get_layerIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
layerIDs_crystal_50 = [x + 1 for x in crystal_features_50[2]]
layerIDs_WSi_50 = [x + 2 for x in get_layerIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)]
layerIDs_PbSi_50 = [x + 25 for x in get_layerIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]
layerIDs_PbScint_50 = [x + 37 for x  in PbScint_features_50[2]]

# %%
hitEnergyDep['50GeV'] = [np.concatenate([eDep_SiPix_50[i], eDep_WSi_50[i], eDep_PbSi_50[i], eDep_PbScint_50[i], eDep_crystal_50[i]]) for i in range(len(eDep_WSi_50))]
xIDs['50GeV'] = [np.concatenate([xIDs_SiPix_50[i], xIDs_WSi_50[i], xIDs_PbSi_50[i], xIDs_PbScint_50[i], xIDs_crystal_50[i]]) for i in range(len(eDep_WSi_50))]
yIDs['50GeV'] = [np.concatenate([yIDs_SiPix_50[i], yIDs_WSi_50[i], yIDs_PbSi_50[i], yIDs_PbScint_50[i], yIDs_crystal_50[i]]) for i in range(len(eDep_WSi_50))]
layerIDs['50GeV'] = [np.concatenate([layerIDs_SiPix_50[i], layerIDs_WSi_50[i], layerIDs_PbSi_50[i], layerIDs_PbScint_50[i], layerIDs_crystal_50[i]]) for i in range(len(eDep_WSi_50))]

# %%
eDep_SiPix_100 = get_eDep(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
eDep_WSi_100 = get_eDep(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
eDep_PbSi_100 = get_eDep(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
eDep_PbScint_100 = PbScint_features_100[3]
eDep_crystal_100 = crystal_features_100[3]

xIDs_SiPix_100 = get_xIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
xIDs_WSi_100 = get_xIDs_WSi(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
xIDs_PbSi_100 = get_xIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
xIDs_PbScint_100 = PbScint_features_100[0]
xIDs_crystal_100 = crystal_features_100[0]

yIDs_SiPix_100 = get_yIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
yIDs_WSi_100 = get_yIDs_WSi(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
yIDs_PbSi_100 = get_yIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
yIDs_PbScint_100 = PbScint_features_100[1]
yIDs_crystal_100 = crystal_features_100[1]

layerIDs_SiPix_100 = get_layerIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
layerIDs_crystal_100 = [x + 1 for x in crystal_features_100[2]]
layerIDs_WSi_100 = [x + 2 for x in get_layerIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)]
layerIDs_PbSi_100 = [x + 25 for x in get_layerIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]
layerIDs_PbScint_100 = [x + 37 for x in PbScint_features_100[2]]

# %%
hitEnergyDep['100GeV'] = [np.concatenate([eDep_SiPix_100[i], eDep_WSi_100[i], eDep_PbSi_100[i], eDep_PbScint_100[i], eDep_crystal_100[i]]) for i in range(len(eDep_WSi_100))]
xIDs['100GeV'] = [np.concatenate([xIDs_SiPix_100[i], xIDs_WSi_100[i], xIDs_PbSi_100[i], xIDs_PbScint_100[i], xIDs_crystal_100[i]]) for i in range(len(eDep_WSi_100))]
yIDs['100GeV'] = [np.concatenate([yIDs_SiPix_100[i], yIDs_WSi_100[i], yIDs_PbSi_100[i], yIDs_PbScint_100[i], yIDs_crystal_100[i]]) for i in range(len(eDep_WSi_100))]
layerIDs['100GeV'] = [np.concatenate([layerIDs_SiPix_100[i], layerIDs_WSi_100[i], layerIDs_PbSi_100[i], layerIDs_PbScint_100[i], layerIDs_crystal_100[i]]) for i in range(len(eDep_WSi_100))]

# %%
eDep_SiPix_150 = get_eDep(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
eDep_WSi_150 = get_eDep(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
eDep_PbSi_150 = get_eDep(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
eDep_PbScint_150 = PbScint_features_150[3]
eDep_crystal_150 = crystal_features_150[3]

xIDs_SiPix_150 = get_xIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
xIDs_WSi_150 = get_xIDs_WSi(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
xIDs_PbSi_150 = get_xIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
xIDs_PbScint_150 = PbScint_features_150[0]
xIDs_crystal_150 = crystal_features_150[0]

yIDs_SiPix_150 = get_yIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
yIDs_WSi_150 = get_yIDs_WSi(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
yIDs_PbSi_150 = get_yIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
yIDs_PbScint_150 = PbScint_features_150[1]
yIDs_crystal_150 = crystal_features_150[1]

layerIDs_SiPix_150 = get_layerIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
layerIDs_crystal_150 = [x + 1 for x in crystal_features_150[2]]
layerIDs_WSi_150 = [x + 2 for x in get_layerIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)]
layerIDs_PbSi_150 = [x + 25 for x in get_layerIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]
layerIDs_PbScint_150 = [x + 37 for x in PbScint_features_150[2]]

# %%
hitEnergyDep['150GeV'] = [np.concatenate([eDep_SiPix_150[i], eDep_WSi_150[i], eDep_PbSi_150[i], eDep_PbScint_150[i], eDep_crystal_150[i]]) for i in range(len(eDep_WSi_150))]
xIDs['150GeV'] = [np.concatenate([xIDs_SiPix_150[i], xIDs_WSi_150[i], xIDs_PbSi_150[i], xIDs_PbScint_150[i], xIDs_crystal_150[i]]) for i in range(len(eDep_WSi_150))]
yIDs['150GeV'] = [np.concatenate([yIDs_SiPix_150[i], yIDs_WSi_150[i], yIDs_PbSi_150[i], yIDs_PbScint_150[i], yIDs_crystal_150[i]]) for i in range(len(eDep_WSi_150))]
layerIDs['150GeV'] = [np.concatenate([layerIDs_SiPix_150[i], layerIDs_WSi_150[i], layerIDs_PbSi_150[i], layerIDs_PbScint_150[i], layerIDs_crystal_150[i]]) for i in range(len(eDep_WSi_150))]

# %%
data_labels = np.concatenate([get_labels(samples[key], nevents) for key in samples])

# %%
with open('hitEnergyDep', 'wb') as handle:
    pickle.dump(hitEnergyDep, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('xIDs', 'wb') as handle:
    pickle.dump(xIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('yIDs', 'wb') as handle:
    pickle.dump(yIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('layerIDs', 'wb') as handle:
    pickle.dump(layerIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data_labels', 'wb') as handle:
    pickle.dump(data_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
with open('hitEnergyDep', 'rb') as handle:
    hitEnergyDep = pickle.load(handle)
with open('xIDs', 'rb') as handle:
    xIDs = pickle.load(handle)
with open('yIDs', 'rb') as handle:
    yIDs = pickle.load(handle)
with open('layerIDs', 'rb') as handle:
    layerIDs = pickle.load(handle)
with open('data_labels', 'rb') as handle:
    data_labels = pickle.load(handle)

# %%
import awkward as ak

hitEnergyDep_all = ak.concatenate(list(hitEnergyDep.values()), axis=0)
xIDs_all = ak.concatenate(list(xIDs.values()), axis=0)
yIDs_all = ak.concatenate(list(yIDs.values()), axis=0)
layerIDs_all = ak.concatenate(list(layerIDs.values()), axis=0)

# %%
from sklearn.model_selection import train_test_split

hitEnergyDep_train, hitEnergyDep_test, xIDs_train, xIDs_test, yIDs_train, yIDs_test, layerIDs_train, layerIDs_test, labels_train, labels_test = train_test_split(hitEnergyDep_all, xIDs_all, yIDs_all, layerIDs_all, data_labels, test_size=0.2, train_size=0.8, random_state=None, shuffle=True)

# %%
features_train = [hitEnergyDep_train, xIDs_train, yIDs_train, layerIDs_train]
features_test = [hitEnergyDep_test, xIDs_test, yIDs_test, layerIDs_test]

# %%
with open('features_train', 'wb') as handle:
    pickle.dump(features_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('features_test', 'wb') as handle:
    pickle.dump(features_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('labels_train', 'wb') as handle:
    pickle.dump(labels_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('labels_test', 'wb') as handle:
    pickle.dump(labels_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
