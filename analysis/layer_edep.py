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
#Get layerID from cellID
def bitExtract(n, k, p):  
    return (((1 << k) - 1)  &  (n >> p))


# %%
def layer_edep(data, branch, count):
    edep = pd.DataFrame()

    for i in range(count):
        index = str(i)
        energies = np.array(data[f"{branch}.energy"][i])
        cellID = np.array(data[f"{branch}.cellID"][i])
        layerID = bitExtract(cellID, 6, 8)
        df = pd.DataFrame({f'{index}': energies, 'layerID': layerID})
        layers = df.groupby("layerID").sum()

        if len(layers) != 0:
            edep = pd.concat([edep,layers], axis=1).replace(np.NaN,0)
            
    return edep


# %%
def component_frac_edep(data, branch, count):
    edep = pd.DataFrame()

    for i in range(count):
        index = str(i)
        energies = np.array(data[f"{branch}.energy"][i])
        cellID = np.array(data[f"{branch}.cellID"][i])
        layerID = bitExtract(cellID, 6, 8)
        df = pd.DataFrame({f'{index}': energies, 'layerID': layerID})
        layers = df.groupby("layerID").sum()
        total_edep = layers.sum()[0]
        frac_edep_layers = layers/total_edep
        
        if len(layers) != 0:
            edep = pd.concat([edep,frac_edep_layers], axis=1).replace(np.NaN,0)
            
    return edep


# %%
def frac_edep(data, branch, count):
    edep = pd.DataFrame()

    for i in range(count):
        index = str(i)
        
        total_energy = np.sqrt(data["MCParticles.momentum.x"][0,0]**2 + data["MCParticles.momentum.y"][0,0]**2 + data["MCParticles.momentum.z"][0,0]**2)

        energies = np.array(data[f"{branch}.energy"][i])
        cellID = np.array(data[f"{branch}.cellID"][i])
        layerID = bitExtract(cellID, 6, 8)
        df = pd.DataFrame({f'{index}': energies, 'layerID': layerID})
        layers = df.groupby("layerID").sum()
        frac_edep_layers = layers/total_energy
        
        if len(layers) != 0:
            edep = pd.concat([edep,frac_edep_layers], axis=1).replace(np.NaN,0)
            
    return edep


# %% [markdown]
# Plots

# %%
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000).iloc[0], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000).iloc[1], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000).iloc[2], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.xlabel('Energy Deposited')
plt.ylabel('Events')
plt.title('Energy Deposition in Pb/Si Layers')
plt.legend(['Layer 1', 'Layer 2', 'Layer 3'])

# %%
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000).iloc[3], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000).iloc[4], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000).iloc[5], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.xlabel('Energy Deposited')
plt.ylabel('Events')
plt.title('Energy Deposition in Pb/Si Layers')
plt.legend(['Layer 4', 'Layer 5', 'Layer 6'])

# %%
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000).iloc[6], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000).iloc[7], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000).iloc[8], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.xlabel('Energy Deposited')
plt.ylabel('Events')
plt.title('Energy Deposition in Pb/Si Layers')
plt.legend(['Layer 7', 'Layer 8', 'Layer 9'])

# %%
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[5], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[6], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[7], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.xlabel('Energy Deposited')
plt.ylabel('Events')
plt.title('Energy Deposition in W/Si Layers')
plt.legend(['Layer 9', 'Layer 10', 'Layer 11'])

# %%
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[8], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[9], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[10], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.xlabel('Energy Deposited')
plt.ylabel('Events')
plt.title('Energy Deposition in W/Si Layers')
plt.legend(['Layer 9', 'Layer 10', 'Layer 11'])

# %%
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[11], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[12], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[13], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.xlabel('Energy Deposited')
plt.ylabel('Events')
plt.title('Energy Deposition in W/Si Layers')
plt.legend(['Layer 12', 'Layer 13', 'Layer 14'])

# %%
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[14], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[15], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[16], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.xlabel('Energy Deposited')
plt.ylabel('Events')
plt.title('Energy Deposition in W/Si Layers')
plt.legend(['Layer 15', 'Layer 16', 'Layer 17'])

# %%
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[17], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[18], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[19], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.xlabel('Energy Deposited')
plt.ylabel('Events')
plt.title('Energy Deposition in W/Si Layers')
plt.legend(['Layer 18', 'Layer 19', 'Layer 20'])

# %%
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[20], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[21], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.hist(layer_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[22], bins=np.linspace(0,0.1,100), histtype='step', alpha=0.7)
plt.xlabel('Energy Deposited')
plt.ylabel('Events')
plt.title('Energy Deposition in W/Si Layers')
plt.legend(['Layer 21', 'Layer 22', 'Layer 23'])

# %%
plt.hist(layer_edep(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[21] - layer_edep(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[20], bins=np.linspace(-0.05,0.05,50), histtype='step')
plt.hist(layer_edep(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[21] - layer_edep(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[20], bins=np.linspace(-0.05,0.05,50), histtype='step')
plt.hist(layer_edep(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[21] - layer_edep(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000).iloc[20], bins=np.linspace(-0.05,0.05,50), histtype='step')
plt.xlabel('Energy Difference')
plt.ylabel('Events')
plt.legend(['10GeV', '50GeV', '150GeV'])
plt.semilogy()
plt.title('Energy Deposition Difference in W/Si Layers 21 and 22')

# %%
frac_10 = frac_edep(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
frac_20 = frac_edep(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
frac_50 = frac_edep(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
frac_100 = frac_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)
frac_150 = frac_edep(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_WSi_Hits", 10000)

# %%
frac_10 = frac_edep(samples['zdc_neutron_10GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)
frac_20 = frac_edep(samples['zdc_neutron_20GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)
frac_50 = frac_edep(samples['zdc_neutron_50GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)
frac_100 = frac_edep(samples['zdc_neutron_100GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)
frac_150 = frac_edep(samples['zdc_neutron_150GeV_10e4.edm4hep.root'], "ZDC_PbSi_Hits", 10000)

# %%
frac_10.mean(axis=1).plot()
frac_20.mean(axis=1).plot()
frac_50.mean(axis=1).plot()
frac_100.mean(axis=1).plot()
frac_150.mean(axis=1).plot()
plt.ylabel('Energy Deposited / Neutron Energy')
plt.legend(['10GeV', '20GeV', '50GeV', '100GeV', '150GeV'])

# %%
frac_10.mean(axis=1).plot()
frac_20.mean(axis=1).plot()
frac_50.mean(axis=1).plot()
frac_100.mean(axis=1).plot()
frac_150.mean(axis=1).plot()
plt.legend(['10GeV', '20GeV', '50GeV', '100GeV', '150GeV'])

# %%
layer_edep(samples["zdc_neutron_10GeV_10e4.edm4hep.root"],"ZDC_WSi_Hits",10000).mean(axis=1).plot()
layer_edep(samples["zdc_neutron_20GeV_10e4.edm4hep.root"],"ZDC_WSi_Hits",10000).mean(axis=1).plot()
layer_edep(samples["zdc_neutron_50GeV_10e4.edm4hep.root"],"ZDC_WSi_Hits",10000).mean(axis=1).plot()
layer_edep(samples["zdc_neutron_100GeV_10e4.edm4hep.root"],"ZDC_WSi_Hits",10000).mean(axis=1).plot()
layer_edep(samples["zdc_neutron_150GeV_10e4.edm4hep.root"],"ZDC_WSi_Hits",10000).mean(axis=1).plot()
plt.ylabel("Energy Deposition")
plt.title("Energy Deposition in W/Si Layers")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.savefig("wsiedep_layers.pdf")

# %%
layer_edep(samples["zdc_neutron_10GeV_10e4.edm4hep.root"],"ZDC_PbSi_Hits",10000).mean(axis=1).plot()
layer_edep(samples["zdc_neutron_20GeV_10e4.edm4hep.root"],"ZDC_PbSi_Hits",10000).mean(axis=1).plot()
layer_edep(samples["zdc_neutron_50GeV_10e4.edm4hep.root"],"ZDC_PbSi_Hits",10000).mean(axis=1).plot()
layer_edep(samples["zdc_neutron_100GeV_10e4.edm4hep.root"],"ZDC_PbSi_Hits",10000).mean(axis=1).plot()
layer_edep(samples["zdc_neutron_150GeV_10e4.edm4hep.root"],"ZDC_PbSi_Hits",10000).mean(axis=1).plot()
plt.ylabel("Energy Deposition")
plt.title("Energy Deposition in Pb/Si Layers")
plt.legend(["10GeV", "20GeV", "50GeV", "100GeV", "150GeV"])
plt.savefig("wsiedep_layers.pdf")
