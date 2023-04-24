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
from bitstring import BitArray
import pandas as pd
import awkward as ak
import pickle

# %%
particleType = 'neutron'
config = 'ip6'
count = '10e4'

# %% [markdown]
# Load Data from ROOT Files

# %%
data_path = f'/home/dmisra/eic/{particleType}_angle_samples/{config}/'

# %%
paths = []

for (path, dirnames, filenames) in os.walk(data_path):
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
    return 3*x, 3*y


# %%
def get_layerIDs(data, branch, events):
    layerID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)

        if len(event_cellID) > 0:
            layerID.append(event_layerID)
        elif len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            layerID.append(np.array([]))
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            layerID.append(np.array([]))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            layerID.append(np.array([]))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            layerID.append(np.array([]))
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            layerID.append(np.array([]))
            
    return layerID


# %%
def get_eDep(data, branch, events):
    hitsEnergy = []
    for i in range(events):
        event_hitsEnergy = np.array(data[f"{branch}.energy"][i])

        if len(event_hitsEnergy) > 0: 
            hitsEnergy.append(event_hitsEnergy)
        elif len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            hitsEnergy.append(np.array([]))
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            hitsEnergy.append(np.array([]))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            hitsEnergy.append(np.array([]))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            hitsEnergy.append(np.array([]))
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            hitsEnergy.append(np.array([]))

    return hitsEnergy


# %%
def get_xIDs(data, branch, events):
    xID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_xID = signedint(bitExtract(event_cellID, 12, 24))

        if len(event_cellID) > 0: 
            xID.append(event_xID)
        elif len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            xID.append(np.array([]))
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            xID.append(np.array([]))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            xID.append(np.array([]))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            xID.append(np.array([]))
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            xID.append(np.array([]))
            
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
        
        if len(event_cellID) > 0:
            xID.append(event_xID)
        elif len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            xID.append(np.array([]))
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            xID.append(np.array([]))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            xID.append(np.array([]))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            xID.append(np.array([]))
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            xID.append(np.array([]))
            
    return xID


# %%
def get_xIDs_SiPix(data, branch, events):
    xID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        event_xID = signedint(bitExtract(event_cellID, 12, 24))
        for i in range(len(event_layerID)):
            event_xID[i] = 0.3 * event_xID[i]
        
        if len(event_cellID) > 0:
            xID.append(event_xID)
        elif len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            xID.append(np.array([]))
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            xID.append(np.array([]))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            xID.append(np.array([]))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            xID.append(np.array([]))
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            xID.append(np.array([]))
            
    return xID


# %%
def get_yIDs(data, branch, events):
    yID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_yID = signedint(bitExtract(event_cellID, 12, 36))

        if len(event_cellID) > 0:
            yID.append(event_yID)
        elif len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            yID.append(np.array([]))
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            yID.append(np.array([]))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            yID.append(np.array([]))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            yID.append(np.array([]))
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            yID.append(np.array([]))
        
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
        
        if len(event_cellID) > 0:
            yID.append(event_yID)
        elif len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            yID.append(np.array([]))
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            yID.append(np.array([]))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            yID.append(np.array([]))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            yID.append(np.array([]))
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            yID.append(np.array([]))

    return yID


# %%
def get_yIDs_SiPix(data, branch, events):
    yID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        event_yID = signedint(bitExtract(event_cellID, 12, 36))
        for i in range(len(event_layerID)):
            event_yID[i] = 0.3 * event_yID[i]
        
        if len(event_cellID) > 0:
            yID.append(event_yID)
        elif len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            yID.append(np.array([]))
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            yID.append(np.array([]))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            yID.append(np.array([]))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            yID.append(np.array([]))
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            yID.append(np.array([]))

    return yID


# %%
def get_PbScint_features(data, branch, events): 
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

        if len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            layerID.append(layerIDs_integrated)
            eDep.append(eDep_integrated)
        elif len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            xID.append(np.array([]))
            yID.append(np.array([]))
            layerID.append(np.array([]))
            eDep.append(np.array([]))
        elif len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            xID.append(np.array([]))
            yID.append(np.array([]))
            layerID.append(np.array([]))
            eDep.append(np.array([]))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            xID.append(np.array([]))
            yID.append(np.array([]))
            layerID.append(np.array([]))
            eDep.append(np.array([]))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            xID.append(np.array([]))
            yID.append(np.array([]))
            layerID.append(np.array([]))
            eDep.append(np.array([]))

    return xID, yID, layerID, eDep



# %%
def get_crystal_features(data, branch, events): 
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

        if len(np.array(data["ZDCEcalHits.cellID"][i])) > 0: 
            xID.append(xIDs_integrated)
            yID.append(yIDs_integrated)
            eDep.append(eDep_integrated)
            layerID.append(np.zeros(len(eDep_integrated)))
        elif len(np.array(data["ZDC_SiliconPix_Hits.cellID"][i])) > 0:
            xID.append(np.array([]))
            yID.append(np.array([]))
            eDep.append(np.array([]))
            layerID.append(np.array([]))
        elif len(np.array(data["ZDC_WSi_Hits.cellID"][i])) > 0: 
            xID.append(np.array([]))
            yID.append(np.array([]))
            eDep.append(np.array([]))
            layerID.append(np.array([]))
        elif len(np.array(data["ZDC_PbSi_Hits.cellID"][i])) > 0: 
            xID.append(np.array([]))
            yID.append(np.array([]))
            eDep.append(np.array([]))
            layerID.append(np.array([]))
        elif len(np.array(data["ZDCHcalHits.cellID"][i])) > 0: 
            xID.append(np.array([]))
            yID.append(np.array([]))
            eDep.append(np.array([]))
            layerID.append(np.array([]))

    return xID, yID, layerID, eDep


# %% [markdown]
# Dataset

# %%
nevents = 10000

# %%
crystal_features = dict()
PbScint_features = dict()

eDep_dict = dict()
xID_dict = dict()
yID_dict = dict()
layerID_dict = dict()

hitEnergyDep = dict()
xIDs = dict()
yIDs = dict()
layerIDs = dict()

# %%
angle_list = ['-4mrad', '00mrad', '04mrad']
component_list = ['SiPix', 'Crystal', 'WSi', 'PbSi', 'PbScint']

# %%
eDep_dict = dict()
xID_dict = dict()
yID_dict = dict()
layerID_dict = dict()

# %%
for component in component_list:
    eDep_dict[f'{component}'] = dict()
    xID_dict[f'{component}'] = dict()
    yID_dict[f'{component}'] = dict()
    layerID_dict[f'{component}'] = dict()

# %%
for angle in angle_list:
    crystal_features[f'{angle}'] = get_crystal_features(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDCEcalHits", nevents)
    PbScint_features[f'{angle}'] = get_PbScint_features(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDCHcalHits", nevents)

# %%
for angle in angle_list:
    eDep_dict['SiPix'][f'{angle}'] = get_eDep(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
    eDep_dict['Crystal'][f'{angle}'] = crystal_features[f'{angle}'][3]
    eDep_dict['WSi'][f'{angle}'] = get_eDep(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_WSi_Hits", nevents)
    eDep_dict['PbSi'][f'{angle}'] = get_eDep(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
    eDep_dict['PbScint'][f'{angle}'] = PbScint_features[f'{angle}'][3]

    xID_dict['SiPix'][f'{angle}'] = get_xIDs_SiPix(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
    xID_dict['Crystal'][f'{angle}'] = crystal_features[f'{angle}'][0]
    xID_dict['WSi'][f'{angle}'] = get_xIDs_WSi(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_WSi_Hits", nevents)
    xID_dict['PbSi'][f'{angle}'] = get_xIDs(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
    xID_dict['PbScint'][f'{angle}'] = PbScint_features[f'{angle}'][0]

    yID_dict['SiPix'][f'{angle}'] = get_yIDs_SiPix(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
    yID_dict['Crystal'][f'{angle}'] = crystal_features[f'{angle}'][1]
    yID_dict['WSi'][f'{angle}'] = get_yIDs_WSi(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_WSi_Hits", nevents)
    yID_dict['PbSi'][f'{angle}'] = get_yIDs(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
    yID_dict['PbScint'][f'{angle}'] = PbScint_features[f'{angle}'][1]

    layerID_dict['SiPix'][f'{angle}'] = get_layerIDs(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_SiliconPix_Hits", nevents)
    layerID_dict['Crystal'][f'{angle}'] = [x + 1 for x in crystal_features[f'{angle}'][2]]
    layerID_dict['WSi'][f'{angle}'] = [x + 2 for x in get_layerIDs(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_WSi_Hits", nevents)]
    layerID_dict['PbSi'][f'{angle}'] = [x + 25 for x in get_layerIDs(samples[f"{config}_{particleType}_{angle}_{count}.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]
    layerID_dict['PbScint'][f'{angle}'] = [x + 37 for x in PbScint_features[f'{angle}'][2]]

# %%
for component in component_list:
    hitEnergyDep[f'{component}'] = ak.concatenate([eDep_dict[f'{component}'][f'{angle}'] for angle in angle_list], axis=0)
    xIDs[f'{component}'] = ak.concatenate([xID_dict[f'{component}'][f'{angle}'] for angle in angle_list], axis=0)
    yIDs[f'{component}'] = ak.concatenate([yID_dict[f'{component}'][f'{angle}'] for angle in angle_list], axis=0)
    layerIDs[f'{component}'] = ak.concatenate([layerID_dict[f'{component}'][f'{angle}'] for angle in angle_list], axis=0)

# %%
xvals, yvals = dict(), dict()

for key in xID_dict:
    xvals[key] = np.concatenate([np.concatenate(xID_dict[key]['-4mrad']), np.concatenate(xID_dict[key]['00mrad']), np.concatenate(xID_dict[key]['04mrad'])])
    yvals[key] = np.concatenate([np.concatenate(yID_dict[key]['-4mrad']), np.concatenate(yID_dict[key]['00mrad']), np.concatenate(yID_dict[key]['04mrad'])])

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.hist2d(xvals['SiPix'], yvals['SiPix'], bins=[np.linspace(-30, 30, 60), np.linspace(-30, 30, 60)], norm='log', cmap=plt.cm.jet);
plt.colorbar();
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.hist2d(xvals['WSi'], yvals['WSi'], bins=[np.linspace(-30, 30, 60), np.linspace(-30, 30, 60)], norm='log', cmap=plt.cm.jet);
plt.colorbar();
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.hist2d(xvals['PbSi'], yvals['PbSi'], bins=[np.linspace(-30, 30, 60), np.linspace(-30, 30, 60)], norm='log', cmap=plt.cm.jet);
plt.colorbar();
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')

# %%
SiPix_xvals, SiPix_yvals = dict(), dict()
for angle in angle_list:
    SiPix_xvals[angle] = np.concatenate(xID_dict['SiPix'][angle])
    SiPix_yvals[angle] = np.concatenate(yID_dict['SiPix'][angle])

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
plots = dict()
for angle in angle_list:
    plt.figure()
    plt.hist2d(SiPix_xvals[angle], SiPix_yvals[angle], bins=[np.linspace(-30, 30, 60), np.linspace(-30, 30, 60)], norm='log', cmap=plt.cm.jet);
    plt.colorbar();
    plt.xlabel('x (cm)');
    plt.ylabel('y (cm)');
plt.show()

# %%
WSi_xvals, WSi_yvals = dict(), dict()
for angle in angle_list:
    WSi_xvals[angle] = np.concatenate(xID_dict['WSi'][angle])
    WSi_yvals[angle] = np.concatenate(yID_dict['WSi'][angle])

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
plots = dict()
for angle in angle_list:
    plt.figure()
    plt.hist2d(WSi_xvals[angle], WSi_yvals[angle], bins=[np.linspace(-30, 30, 60), np.linspace(-30, 30, 60)], norm='log', cmap=plt.cm.jet);
    plt.colorbar();
    plt.xlabel('x (cm)');
    plt.ylabel('y (cm)');
plt.show()


# %%
def get_l1_com(angle):
    l1_com_list = []
    for i in range(len(xID_dict['SiPix'][f'{angle}'])):
        hit_coordinates = [xID_dict['SiPix'][f'{angle}'][i], yID_dict['SiPix'][f'{angle}'][i], layerID_dict['SiPix'][f'{angle}'][i]]
        l1_com = np.average(hit_coordinates, weights=eDep_dict['SiPix'][f'{angle}'][i], axis=1)
        l1_com_list.append(l1_com)
    return l1_com_list


# %%
def get_last5_com(angle):
    last5_com_list = []
    for i in range(len(layerID_dict['PbScint'][f'{angle}'])):
        lastLayer = max(layerID_dict['PbScint'][f'{angle}'][i])
        last5 = [lastLayer, lastLayer-1, lastLayer-2, lastLayer-3, lastLayer-4]
        last5_xIDs = []
        last5_yIDs = []
        last5_layerIDs = []
        last5_eDep = []
        for j in range(len(layerID_dict['PbScint'][f'{angle}'][i])):
            if layerID_dict['PbScint'][f'{angle}'][i][j] in last5:
                last5_xIDs.append(xID_dict['PbScint'][f'{angle}'][i][j])
                last5_yIDs.append(yID_dict['PbScint'][f'{angle}'][i][j])
                last5_layerIDs.append(layerID_dict['PbScint'][f'{angle}'][i][j])
                last5_eDep.append(eDep_dict['PbScint'][f'{angle}'][i][j])
        last5_com = np.average([last5_xIDs, last5_yIDs, last5_layerIDs], weights=last5_eDep, axis=1)
        last5_com_list.append(last5_com)
    return
