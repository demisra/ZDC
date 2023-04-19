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
import numpy as np
import pickle
import matplotlib.pyplot as plt
import awkward as ak

# %%
with open('features_train', 'rb') as handle:
    features_train = pickle.load(handle)

with open('features_test', 'rb') as handle:
    features_test = pickle.load(handle)

with open('labels_train', 'rb') as handle:
    labels_train = pickle.load(handle)

with open('labels_test', 'rb') as handle:
    labels_test = pickle.load(handle)

# %% [markdown]
# PyTorch Geometric

# %%
import torch
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph, GCNConv, global_add_pool

# %%
torch.cuda.is_available()


# %%
class ZDCDataset(Dataset):
    def __init__(self, features, labels, knn_k):
        super(ZDCDataset, self).__init__()
        
        self.knn_k = knn_k
        
        self.energy = features[0]
        self.xID = features[1]
        self.yID = features[2]
        self.layerID = features[3]
        self.label = labels

    def len(self):
        return len(self.energy)

    def get(self, idx):
        
        energy = torch.tensor(self.energy[idx]).to(torch.float32)
        xID = torch.tensor(self.xID[idx]).to(torch.float32)
        yID = torch.tensor(self.yID[idx]).to(torch.float32)
        layerID = torch.tensor(self.layerID[idx]).to(torch.float32)
        
        label = torch.tensor(self.label[idx]).to(torch.float32)
        
        x = torch.stack([energy, xID, yID, layerID], axis=-1)
        
        #construct knn graph from (x, y, z) coordinates
        edge_index = knn_graph(x[:, [1,3]], k=self.knn_k, num_workers=32)
        
        data = Data(
            x = x,
            y = label,
            edge_index = edge_index
        )
        
        return data


# %%
dataset = ZDCDataset(features_train, labels_train, knn_k=32)
loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=32)

# %%
ibatch = 0
for data_batched in loader:
    print(ibatch, data_batched.x.shape, data_batched.y)
    ibatch += 1
    if ibatch>5:
        break

# %% [markdown]
# GCN Model

# %%
from torch_geometric.nn import GCNConv, global_add_pool

class Net(torch.nn.Module):
    def __init__(self, num_node_features=4):
        super(Net, self).__init__()
        
        #(4 -> N)
        self.conv1 = GCNConv(num_node_features, 512)
        
        #(N -> 1)
        self.output = torch.nn.Linear(512, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        #add a batch index, in in case we are running on a single graph
        if not hasattr(data, "batch"):
            data.batch = torch.zeros(len(x), dtype=torch.int64).to(x.device)
        
        #Transform the nodes with the graph convolution
        transformed_nodes = self.conv1(x, edge_index)
        transformed_nodes = torch.nn.functional.elu(transformed_nodes)
        
        #Sum up all the node vectors in each graph according to the batch index
        per_graph_aggregation = global_add_pool(transformed_nodes, data.batch)
        
        #For each graph,
        #predict the output based on the total vector
        #from the previous aggregation step
        output = self.output(per_graph_aggregation)
        return output


# %%
net = Net()

# %%
net(data_batched)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=1e-2, threshold_mode='rel', patience=10)

# %%
model.train()
losses_train = []

for epoch in range(200):
    
    loss_train_epoch = []
    
    for data_batch in loader:
        
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        out = model(data_batch)
        loss = nn.functional.mse_loss(out[:, 0], data_batch.y)
        loss.backward()
        loss_train_epoch.append(loss.item())
        optimizer.step()

    loss_train_epoch = np.mean(loss_train_epoch)
    learning_rate_epoch = scheduler.optimizer.param_groups[0]['lr']
    losses_train.append(loss_train_epoch)
    print(epoch, loss_train_epoch, learning_rate_epoch)

    scheduler.step(loss_train_epoch)

    if epoch % 100 == 0:
        torch.save(obj=model.state_dict(), f=f"/home/dmisra/eic/gnn_state_dict_{epoch}")

# %%
plt.plot(losses_train, label="training")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.semilogy()

# %%
ievent = 42
data = dataset.get(ievent).to(device)
embedded_nodes = model.conv1(data.x, data.edge_index)

# %%
test_dataset = ZDCDataset(features_test, labels_test, knn_k=32)
test_loader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

# %%
model.load_state_dict(torch.load('/home/dmisra/eic/gnn_state_dict_100'))

# %% [markdown]
# Predictions

# %%
from scipy.stats import norm
from scipy.optimize import curve_fit

# %%
torch.cuda.empty_cache()

# %%
#Set the model in evaluation mode
model.eval()

#Setup the inference mode context manager
with torch.inference_mode():
  y_preds_batched=[]
  for batch in test_loader:
    y_preds_batched.extend(model(batch.cuda()))
  y_preds = torch.concat([y.cpu() for y in y_preds_batched])

# %%
plt.hist(y_preds,100,histtype='step')
plt.xlabel('Energy (GeV)')
plt.ylabel('Count')
plt.title('Predicted Energy Distribution')


# %%
def tensorIntersect(t1, t2):
    a = set((tuple(i) for i in t1.numpy()))
    b = set((tuple(i) for i in t2.numpy()))
    c = a.intersection(b)
    tensorform = torch.from_numpy(np.array(list(c)))

    return tensorform


# %%
#Set the model in evaluation mode
model.eval()

#Setup the inference mode context manager
with torch.inference_mode():
  y_preds_200GeV = model(features_150GeV)
  y_preds_100GeV = model(features_100GeV)
  y_preds_50GeV = model(features_50GeV)
  y_preds_20GeV = model(features_20GeV)
  y_preds_10GeV = model(features_10GeV)

# %%
peak_preds = norm.fit(y_preds_10GeV)[0], norm.fit(y_preds_20GeV)[0], norm.fit(y_preds_50GeV)[0], norm.fit(y_preds_100GeV)[0], norm.fit(y_preds_150GeV)[0]
true_peaks = [10,20,50,100,150]
peak_preds

# %%
plt.scatter(true_peaks,peak_preds)
plt.xlabel('Particle Energy (GeV)')
plt.ylabel('Reconstructed Energy (GeV)')
plt.plot(np.arange(1,201),np.arange(1,201))
plt.title('Linearity')


# %%
#Get energy resolution from distribution of predictions
def res(preds,energy):
    return norm.fit(preds)[1]/energy

energy_list = [200,100,50,10]
resolutions = res(y_preds_200GeV,200), res(y_preds_100GeV,100), res(y_preds_50GeV,50), res(y_preds_10GeV,10)


# %%
#Curve fit for energy resolution as a function of energy
def f(E,a):
    return a/np.sqrt(E)

popt, pcov = curve_fit(f, energy_list, resolutions)

# %%
popt, pcov

# %%
plt.plot(range(200),f(range(1,201),popt[0]))
plt.scatter(energy_list,resolutions)
plt.xlabel('Energy (GeV)')
plt.ylabel('Resolution')
plt.title('Energy Resolution')
