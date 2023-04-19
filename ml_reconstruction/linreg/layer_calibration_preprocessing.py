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
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import os
import uproot as ur
import numpy as np
import pandas as pd

# %%
paths = []

for (path, dirnames, filenames) in os.walk('/home/dmisra/eic/zdc_neutron_samples'):
    paths.extend(os.path.join(path, name) for name in filenames)

# %%
samples = {}

for path in paths:
    with ur.open(path) as file:
       tree = file["events"]
       samples[os.path.basename(f'{path}')] = tree.arrays()


# %%
def bitExtract(n, k, p):  
    return (((1 << k) - 1)  &  (n >> p))


# %%
def get_labels(data, count):
    energy_labels = []
    for i in range(count):
        index = str(i)
        label = np.sqrt(data["MCParticles.momentum.x"][0,0]**2 + data["MCParticles.momentum.y"][0,0]**2 + data["MCParticles.momentum.z"][0,0]**2)
        energy_labels.append(label)
    
    return energy_labels


# %%
def sipix_edep(data, count):
    edep = pd.DataFrame(index=np.arange(1,2))
    for i in range(count):
        index = str(i)
        energies = np.array(data['ZDC_SiliconPix_Hits.energy'][i])
        cellID = np.array(data['ZDC_SiliconPix_Hits.cellID'][i])
        layerID = bitExtract(cellID, 6, 8)
        df = pd.DataFrame({f'{index}': energies, 'layerID': layerID})
        layers = df.groupby("layerID").sum()
        edep = pd.concat([edep,layers[f'{index}']],axis=1)

    return edep.T.replace(np.NaN,0)


# %%
def crystal_edep(data, count):
    edep = pd.DataFrame(index=np.arange(2,3))
    for i in range(count):
        index = str(i)
        energies = np.array(data["ZDCEcalHits.energy"][i])
        cellID = np.array(data["ZDCEcalHits.cellID"][i])
        crystalID = bitExtract(cellID, 12, 10)
        for i in range(len(crystalID)):
            crystalID[i] = 2
        df = pd.DataFrame({f'{index}': energies, 'crystalID': crystalID})
        crystals = df.groupby("crystalID").sum()
        edep = pd.concat([edep,crystals[f'{index}']],axis=1)
        
    return edep.T.replace(np.NaN,0)


# %%
def imaging_edep(data, count):
    edep = pd.DataFrame(index=np.arange(3,26))
    for i in range(count):
        index = str(i)
        energies = np.array(data["ZDC_WSi_Hits.energy"][i])
        cellID = np.array(data["ZDC_WSi_Hits.cellID"][i])
        layerID = bitExtract(cellID, 6, 8) + 2
        df = pd.DataFrame({f'{index}': energies, 'layerID': layerID})
        layers = df.groupby("layerID").sum()
        edep = pd.concat([edep,layers[f'{index}']],axis=1)
            
    return edep.T.replace(np.NaN,0)


# %%
def pbsi_edep(data, count):
    edep = pd.DataFrame(index=np.arange(26,38))
    for i in range(count):
        index = str(i)
        energies = np.array(data["ZDC_PbSi_Hits.energy"][i])
        cellID = np.array(data["ZDC_PbSi_Hits.cellID"][i])
        layerID = bitExtract(cellID, 6, 8) + 25
        df = pd.DataFrame({f'{index}': energies, 'layerID': layerID})
        layers = df.groupby("layerID").sum()
        edep = pd.concat([edep,layers[f'{index}']],axis=1)
            
    return edep.T.replace(np.NaN,0)


# %%
def pbscint_edep(data, count):
    edep = pd.DataFrame(index=np.arange(38,68))
    for i in range(count):
        index = str(i)
        energies = np.array(data["ZDCHcalHits.energy"][i])
        cellID = np.array(data["ZDCHcalHits.cellID"][i])
        layerID = bitExtract(cellID, 6, 8) + 37
        df = pd.DataFrame({f'{index}': energies, 'layerID': layerID})
        layers = df.groupby("layerID").sum()
        edep = pd.concat([edep,layers[f'{index}']],axis=1)
            
    return edep.T.replace(np.NaN,0)


# %% [markdown]
# Generate Datasets

# %%
sipix_data = pd.concat([sipix_edep(samples[key], 10000) for key in samples])

# %%
crystal_data = pd.concat([crystal_edep(samples[key], 10000) for key in samples])

# %%
wsi_data = pd.concat([imaging_edep(samples[key], 10000) for key in samples])

# %%
pbsi_data = pd.concat([pbsi_edep(samples[key], 10000) for key in samples])

# %%
pbscint_data = pd.concat([pbscint_edep(samples[key], 10000) for key in samples])

# %%
data_labels = np.concatenate([get_labels(samples[key], 10000) for key in samples])
np.savetxt('/home/dmisra/eic/zdc_data/dnn_labels.csv', data_labels)

# %%
data_features = pd.concat([sipix_data, crystal_data, wsi_data, pbsi_data, pbscint_data], axis=1)
data_features.to_csv('/home/dmisra/eic/zdc_data/dnn_features.csv', index=False)

# %%
data_features_10GeV = data_features.iloc[0:10000]
data_features_10GeV.to_csv('/home/dmisra/eic/zdc_data/dnn_features_10GeV.csv', index=False)

data_features_20GeV = data_features.iloc[10000:20000]
data_features_20GeV.to_csv('/home/dmisra/eic/zdc_data/dnn_features_20GeV.csv', index=False)

data_features_50GeV = data_features.iloc[20000:30000]
data_features_50GeV.to_csv('/home/dmisra/eic/zdc_data/dnn_features_50GeV.csv', index=False)

data_features_100GeV = data_features.iloc[30000:40000]
data_features_100GeV.to_csv('/home/dmisra/eic/zdc_data/dnn_features_100GeV.csv', index=False)

data_features_150GeV = data_features.iloc[40000:50000]
data_features_150GeV.to_csv('/home/dmisra/eic/zdc_data/dnn_features_150GeV.csv', index=False)
