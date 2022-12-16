#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import uproot as ur
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bitstring import BitArray


# In[2]:


paths = []

for (path, dirnames, filenames) in os.walk('/mnt/scratch3/dmisra/zdcdata_current/'):
    paths.extend(os.path.join(path, name) for name in filenames)


# In[3]:


samples = {}

for path in paths:
    with ur.open(path) as file:
       tree = file["events"]
       samples[os.path.basename(f'{path}')] = tree.arrays()


# Bit Manipulation

# In[ ]:


def bitExtract(pattern, length, position):  
    return ((1 << length) - 1)  &  (pattern >> (position - 1))


# In[ ]:


def signedint(xbits):
    x_int = []
    x_bin = np.vectorize(np.binary_repr, otypes=[str])(xbits, width=12)
    for bits in x_bin:
        if bits[0] == 0:
             x_int.append(BitArray(bin=bits).int)
        else:
            x_int.append(-BitArray(bin=bits[1:]).int)
    return np.array(x_int)


# Energy Deposition per Cell

# In[ ]:


branches = ['ZDC_SiliconPix_Hits', 'ZDC_WSi_Hits', 'ZDC_PbSi_Hits', 'ZDCHcalHits']


# In[ ]:


def cell_features(data, count, branch):
    event_features = []
    energy_labels = []
    
    for i in range(count):
        label = np.sqrt(data["MCParticles.momentum.x"][0,0]**2 + data["MCParticles.momentum.y"][0,0]**2 + data["MCParticles.momentum.z"][0,0]**2)
        energy_labels.append(label)
        
        energies = np.array(data[f"{branch}.energy"][i])
        cellID = np.array(data[f"{branch}.cellID"][i])
        layerID = bitExtract(cellID, 6, 9)
        idx = signedint(bitExtract(cellID, 11, 25))
        idy = signedint(bitExtract(cellID, 11, 37))
        
        df = pd.DataFrame((zip(idx, idy, layerID, energies)), columns=['idx', 'idy', 'layerID', 'energy'])
        df_grouped = df.groupby(["layerID", "idx", "idy"])['energy'].sum().reset_index().replace(np.NaN, 0.)

        event_features.append(np.array(df_grouped))
        
    return event_features, energy_labels


# Graph Convolutional Network

# In[ ]:


import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import knn_graph, GCNConv, global_add_pool


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device
torch.manual_seed(42)


# In[ ]:


merged_data = [np.concatenate(list(samples.values()))]


# In[ ]:


data_features = cell_features(merged_data[0], 40000, branches[1])[0]


# In[15]:


data_labels = cell_features(merged_data[0], 40000, branches[1])[1]
np.savetxt('/home/dmisra/eic/zdc_data/gnn_labels.csv', data_labels)


# In[ ]:


data_labels = np.loadtxt('/home/dmisra/eic/zdc_data/gnn_labels.csv', delimiter=',')


# In[ ]:


labels = torch.from_numpy(data_labels).float()


# In[14]:


def graph_set(data):
    graph_set = []
    for i in range(40000):
        tensor = torch.from_numpy(data[i].astype(float))
        graph = knn_graph(tensor[:, [0,1,2]], 8)
        graph_set.append(graph)

    return graph_set


# In[16]:


graphs = graph_set(data_features)