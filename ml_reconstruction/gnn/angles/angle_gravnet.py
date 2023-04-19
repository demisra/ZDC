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
import matplotlib.pyplot as plt
import pickle
import awkward as ak

# %%
with open('/home/dmisra/eic/zdc/ml_reconstruction/data/features_train', 'rb') as handle:
    features_train = pickle.load(handle)

with open('/home/dmisra/eic/zdc/ml_reconstruction/data/features_test', 'rb') as handle:
    features_test = pickle.load(handle)

with open('/home/dmisra/eic/zdc/ml_reconstruction/data/labels_train', 'rb') as handle:
    labels_train = pickle.load(handle)

with open('/home/dmisra/eic/zdc/ml_reconstruction/data/labels_test', 'rb') as handle:
    labels_test = pickle.load(handle)

# %%
config = 'ip6'
particleType = 'neutron'

# %% [markdown]
# PyTorch Geometric

# %%
import torch
from torch import nn
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph, GCNConv, global_add_pool

# %%
torch.cuda.is_available()


# %%
class ZDCDataset(Dataset):
    def __init__(self, features, labels):
        super(ZDCDataset, self).__init__()
        
        self.energy_SiPix = features['SiPix'][0]
        self.xID_SiPix = features['SiPix'][1]
        self.yID_SiPix = features['SiPix'][2]
        self.layerID_SiPix = features['SiPix'][3]

        self.energy_Crystal = features['Crystal'][0]
        self.xID_Crystal = features['Crystal'][1]
        self.yID_Crystal = features['Crystal'][2]
        self.layerID_Crystal = features['Crystal'][3]

        self.energy_WSi = features['WSi'][0]
        self.xID_WSi = features['WSi'][1]
        self.yID_WSi = features['WSi'][2]
        self.layerID_WSi = features['WSi'][3]

        self.energy_PbSi = features['PbSi'][0]
        self.xID_PbSi = features['PbSi'][1]
        self.yID_PbSi = features['PbSi'][2]
        self.layerID_PbSi = features['PbSi'][3]

        self.energy_PbScint = features['PbScint'][0]
        self.xID_PbScint = features['PbScint'][1]
        self.yID_PbScint = features['PbScint'][2]
        self.layerID_PbScint = features['PbScint'][3]

        self.label = labels

    def len(self):
        return len(self.label)

    def get(self, idx):
        
        energy_SiPix = torch.tensor(self.energy_SiPix[idx]).to(torch.float32)
        xID_SiPix = torch.tensor(self.xID_SiPix[idx]).to(torch.float32)
        yID_SiPix = torch.tensor(self.yID_SiPix[idx]).to(torch.float32)
        layerID_SiPix = torch.tensor(self.layerID_SiPix[idx]).to(torch.float32)

        energy_Crystal = torch.tensor(self.energy_Crystal[idx]).to(torch.float32)
        xID_Crystal = torch.tensor(self.xID_Crystal[idx]).to(torch.float32)
        yID_Crystal = torch.tensor(self.yID_Crystal[idx]).to(torch.float32)
        layerID_Crystal = torch.tensor(self.layerID_Crystal[idx]).to(torch.float32)

        energy_WSi = torch.tensor(self.energy_WSi[idx]).to(torch.float32)
        xID_WSi = torch.tensor(self.xID_WSi[idx]).to(torch.float32)
        yID_WSi = torch.tensor(self.yID_WSi[idx]).to(torch.float32)
        layerID_WSi = torch.tensor(self.layerID_WSi[idx]).to(torch.float32)

        energy_PbSi = torch.tensor(self.energy_PbSi[idx]).to(torch.float32)
        xID_PbSi = torch.tensor(self.xID_PbSi[idx]).to(torch.float32)
        yID_PbSi = torch.tensor(self.yID_PbSi[idx]).to(torch.float32)
        layerID_PbSi = torch.tensor(self.layerID_PbSi[idx]).to(torch.float32)

        energy_PbScint = torch.tensor(self.energy_PbScint[idx]).to(torch.float32)
        xID_PbScint = torch.tensor(self.xID_PbScint[idx]).to(torch.float32)
        yID_PbScint = torch.tensor(self.yID_PbScint[idx]).to(torch.float32)
        layerID_PbScint = torch.tensor(self.layerID_PbScint[idx]).to(torch.float32)
        
        label = torch.tensor(self.label[idx]).to(torch.float32)
        
        x_SiPix = torch.stack([energy_SiPix, xID_SiPix, yID_SiPix, layerID_SiPix], axis=-1)
        x_Crystal = torch.stack([energy_Crystal, xID_Crystal, yID_Crystal, layerID_Crystal], axis=-1)
        x_WSi = torch.stack([energy_WSi, xID_WSi, yID_WSi, layerID_WSi], axis=-1)
        x_PbSi = torch.stack([energy_PbSi, xID_PbSi, yID_PbSi, layerID_PbSi], axis=-1)
        x_PbScint = torch.stack([energy_PbScint, xID_PbScint, yID_PbScint, layerID_PbScint], axis=-1)
        
        data = HeteroData()
        
        data['SiPix'].x = x_SiPix
        data['Crystal'].x = x_Crystal
        data['WSi'].x = x_WSi
        data['PbSi'].x = x_PbSi
        data['PbScint'].x = x_PbScint
        for key in ['SiPix', 'Crystal', 'WSi', 'PbSi', 'PbScint']: 
            data[f'{key}'].y = label
            data[f'{key}'].num_nodes = len(data[f'{key}'].x)
        
        return data


# %%
dataset = ZDCDataset(features_train, labels_train)
loader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=32)

# %%
test_dataset = ZDCDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=32, pin_memory = True, num_workers=32)

# %% [markdown]
# GCN Model

# %%
from torch_geometric.nn import GravNetConv, global_add_pool


# %%
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.lin1 = GravNetConv(in_channels=-1, out_channels=64, space_dimensions=3, propagate_dimensions=1, k=16)

        self.conv1 = GravNetConv(in_channels=-1, out_channels=64, space_dimensions=3, propagate_dimensions=1, k=16)

        self.conv2 = GravNetConv(in_channels=-1, out_channels=64, space_dimensions=3, propagate_dimensions=1, k=16)

        self.conv3 = GravNetConv(in_channels=-1, out_channels=64, space_dimensions=3, propagate_dimensions=1, k=16)

        self.conv4 = GravNetConv(in_channels=-1, out_channels=64, space_dimensions=3, propagate_dimensions=1, k=16)

        self.output = torch.nn.Linear(320, 1)

    def forward(self, data):

        x_SiPix = data['SiPix'].x
        transformed_nodes_SiPix = self.conv1(x_SiPix)
        transformed_nodes_SiPix = torch.nn.functional.elu(transformed_nodes_SiPix)
        per_graph_aggregation_SiPix = global_add_pool(transformed_nodes_SiPix, data['SiPix'].batch)

        x_Crystal = data['Crystal'].x
        transformed_nodes_Crystal = self.conv1(x_Crystal)
        transformed_nodes_Crystal = torch.nn.functional.elu(transformed_nodes_Crystal)
        per_graph_aggregation_Crystal = global_add_pool(transformed_nodes_Crystal, data['Crystal'].batch)

        x_WSi = data['WSi'].x
        transformed_nodes_WSi = self.conv2(x_WSi)
        transformed_nodes_WSi = torch.nn.functional.elu(transformed_nodes_WSi)
        per_graph_aggregation_WSi = global_add_pool(transformed_nodes_WSi, data['WSi'].batch)

        x_PbSi = data['PbSi'].x
        transformed_nodes_PbSi = self.conv3(x_PbSi)
        transformed_nodes_PbSi = torch.nn.functional.elu(transformed_nodes_PbSi)
        per_graph_aggregation_PbSi = global_add_pool(transformed_nodes_PbSi, data['PbSi'].batch)

        x_PbScint = data['PbScint'].x
        transformed_nodes_PbScint = self.conv4(x_PbScint)
        transformed_nodes_PbScint = torch.nn.functional.elu(transformed_nodes_PbScint)
        per_graph_aggregation_PbScint = global_add_pool(transformed_nodes_PbScint, data['PbScint'].batch)

        stack_concat = torch.cat((per_graph_aggregation_SiPix, per_graph_aggregation_Crystal, per_graph_aggregation_WSi, per_graph_aggregation_PbSi, per_graph_aggregation_PbScint), axis=1)

        output = self.output(stack_concat)

        return output


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=1e-3, threshold_mode='rel', patience=10)

# %%
model.train()
losses_train = []

for epoch in range(500):
    
    loss_train_epoch = []
    
    for data_batch in loader:
        
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        out = model(data_batch)
        loss = nn.functional.mse_loss(out[:,0], data_batch['SiPix'].y)
        loss.backward()
        loss_train_epoch.append(loss.item())
        optimizer.step()

    loss_train_epoch = np.mean(loss_train_epoch)
    learning_rate_epoch = scheduler.optimizer.param_groups[0]['lr']
    losses_train.append(loss_train_epoch)
    print(epoch, loss_train_epoch, learning_rate_epoch)

    scheduler.step(loss_train_epoch)
    

    if epoch % 100 == 0:
        torch.save(obj=model.state_dict(), f=f"/home/dmisra/eic/zdc/ml_reconstruction/gnn/angles/{config}_{particleType}_state_dict_{epoch}")

        #Set the model in evaluation mode
    model.eval()

    #Setup the inference mode context manager
    with torch.inference_mode():
        y_preds_batched=[]
        for batch in test_loader:
            y_preds_batched.extend(model(batch.cuda()))
        y_preds = torch.tensor([y.cpu() for y in y_preds_batched]).numpy()

# %%
plt.plot(losses_train, label="training")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.semilogy()

# %%
#model.load_state_dict(torch.load('/home/dmisra/eic/zdc/ml_reconstruction/gnn/angles/ip6_neutron_angle_state_dict'))

# %% [markdown]
# Predictions

# %%
from scipy.stats import norm
from scipy.optimize import curve_fit

# %%
#Set the model in evaluation mode
model.eval()

#Setup the inference mode context manager
with torch.inference_mode():
  y_preds_batched=[]
  for batch in test_loader:
    y_preds_batched.extend(model(batch.cuda()))
  y_preds = torch.tensor([y.cpu() for y in y_preds_batched]).numpy()

labels_batched=[]
for batch in test_loader:
  labels_batched.extend(batch['SiPix'].y)
labels_all = [y.cpu().numpy() for y in labels_batched]

# %%
plt.hist(labels_all,100,histtype='step')
plt.hist(y_preds,100,histtype='step')
plt.xlabel('Angle (mrad)')
plt.ylabel('Count')
plt.title('Predicted Angle Distribution')

# %%
y,x,_ = plt.hist(y_preds, 100, histtype='step', label='data')
x =(x[1:]+x[:-1])/2 


# %%
def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def biimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def trimodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)


# %%
expected_bi = (0, .1, 500, 4, .1, 500)
expected_tri = (0, .1, 200, 2, .1, 200, 4, .1, 200)
params, cov = curve_fit(trimodal, x, y, expected_tri)
sigma = np.sqrt(np.diag(cov))
x_fit = np.linspace(x.min(), x.max(), 500)
plt.plot(x_fit, trimodal(x_fit, *params), color='red', lw=1, label='model');

# %%
plt.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='distribution 1');
plt.plot(x_fit, gauss(x_fit, *params[3:6]), color='red', lw=1, ls=":", label='distribution 2');
plt.plot(x_fit, gauss(x_fit, *params[6:]), color='red', lw=1, ls="-", label='distribution 3');
plt.hist(y_preds, 100, histtype='step', label='data');
plt.xlabel('Angle (mrad)')
plt.ylabel('Count')
plt.title('Predicted Angle Distribution')

# %%
print(params[:3])
print(params[3:])

# %%
peak_preds = [params[0], params[3]]
true_peaks = [0., 4.]
peak_preds

# %%
print(sigma[0])
print(sigma[3])

# %%
plt.scatter(true_peaks,peak_preds)
plt.xlabel('Particle Angle (mrad)')
plt.ylabel('Reconstructed Angle (mrad)')
plt.plot(np.arange(0,5),np.arange(0,5))
plt.title('Linearity')
