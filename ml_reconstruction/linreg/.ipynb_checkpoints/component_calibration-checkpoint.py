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
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy.optimize import curve_fit
import torch
from torch import nn

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
# Total Energy Deposition in All ZDC Components

# %%
def components_edep(data, count):
    SiPix_edep = []
    Crystal_edep = []
    WSi_edep = []
    PbSi_edep = []
    PbScint_edep = []
    energylabels = []
    
    for i in range(count):
        SiPix_energies = np.array(data["ZDC_SiliconPix_Hits.energy"][i])
        SiPix_edep.append(sum(SiPix_energies))

        Crystal_energies = np.array(data["ZDCEcalHits.energy"][i])
        Crystal_edep.append(sum(Crystal_energies))

        WSi_energies = np.array(data["ZDC_WSi_Hits.energy"][i])
        WSi_edep.append(sum(WSi_energies))

        PbSi_energies = np.array(data["ZDC_PbSi_Hits.energy"][i])
        PbSi_edep.append(sum(PbSi_energies))

        PbScint_energies = np.array(data["ZDCHcalHits.energy"][i])
        PbScint_edep.append(sum(PbScint_energies))

        label = np.sqrt(data["MCParticles.momentum.x"][0,0]**2 + data["MCParticles.momentum.y"][0,0]**2 + data["MCParticles.momentum.z"][0,0]**2)
        energylabels.append(label)

    return pd.DataFrame([SiPix_edep, Crystal_edep, WSi_edep, PbSi_edep, PbScint_edep, energylabels])


# %%
#Data for linear regression
data = [components_edep(samples[key],10000) for key in samples]
data_df = pd.concat(data,axis=1)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
device
torch.manual_seed(42)

# %% [markdown]
# Linear Regression

# %%
data_tensor_10GeV = torch.from_numpy(data[0].values).T.float()
data_tensor_20GeV = torch.from_numpy(data[1].values).T.float()
data_tensor_50GeV = torch.from_numpy(data[2].values).T.float()
data_tensor_100GeV = torch.from_numpy(data[3].values).T.float()
data_tensor_150GeV = torch.from_numpy(data[4].values).T.float()

# %%
data_tensor = torch.from_numpy(data_df.values).T.float()


# %%
def features(tensor):
    return tensor[:,:5]

def labels(tensor):
    return tensor[:,5].unsqueeze(dim=1)


# %%
x = features(data_tensor)
y = labels(data_tensor)

# %%
#Split train/test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

input_size = 5
output_size = 1


# %%
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self,x):
        out = self.linear(x) 
        return out


# %%
model_0 = LinearRegression(input_size,output_size)

# %%
model_0.load_state_dict(torch.load('/home/dmisra/eic/model_0')())

# %%
for layer in model_0.modules():
    if isinstance(layer, nn.Linear):
         layer.weight.data.fill_(1)
         layer.bias.data.fill_(0)

# %%
learning_rate = 0.005
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model_0.parameters(),lr=learning_rate)

# %%
#Set number of epochs
epochs = 500000

#Create lists to track loss values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ###Training
    model_0.train()
    y_pred = model_0(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ###Testing
    model_0.eval()

    with torch.inference_mode():
        test_pred = model_0(x_test)
        test_loss = loss_fn(test_pred, y_test.type(torch.float))

        if epoch % 100 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MSE Train Loss: {loss} | MSE Test Loss: {test_loss}")


# %%
# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and Test Loss Curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

# %%
# Find the model's learned parameters
print(model_0.state_dict())

# %% [markdown]
# Predictions

# %%
#Set the model in evaluation mode
model_0.eval()

#Setup the inference mode context manager
with torch.inference_mode():
  y_preds = model_0(x_test)

plt.hist(y_preds[:,0].numpy(),100,histtype='step')
plt.xlabel('Energy (GeV)')
plt.ylabel('Count')
plt.title('Predicted Energy Distribution')
plt.savefig("linreg_distribution.pdf")


# %%
def tensorIntersect(t1, t2):
    a = set((tuple(i) for i in t1.numpy()))
    b = set((tuple(i) for i in t2.numpy()))
    c = a.intersection(b)
    tensorform = torch.from_numpy(np.array(list(c)))

    return tensorform


# %%
test_150GeV = tensorIntersect(features(data_tensor_150GeV), x_test)
test_100GeV = tensorIntersect(features(data_tensor_100GeV), x_test)
test_50GeV = tensorIntersect(features(data_tensor_50GeV), x_test)
test_20GeV = tensorIntersect(features(data_tensor_20GeV), x_test)
test_10GeV = tensorIntersect(features(data_tensor_10GeV), x_test)

# %%
#Set the model in evaluation mode
model_0.eval()

#Setup the inference mode context manager
with torch.inference_mode():
  y_preds_150GeV = model_0(test_150GeV)
  y_preds_100GeV = model_0(test_100GeV)
  y_preds_50GeV = model_0(test_50GeV)
  y_preds_20GeV = model_0(test_20GeV)
  y_preds_10GeV = model_0(test_10GeV)

# %%
peak_preds = norm.fit(y_preds_10GeV)[0], norm.fit(y_preds_20GeV)[0], norm.fit(y_preds_50GeV)[0], norm.fit(y_preds_100GeV)[0], norm.fit(y_preds_150GeV)[0]
true_peaks = [10,20,50,100,150]
peak_preds

# %%
plt.scatter(true_peaks,peak_preds)
plt.xlabel('Particle Energy (GeV)')
plt.ylabel('Reconstructed Energy (GeV)')
plt.plot(np.arange(1,151),np.arange(1,151))
plt.title('Linearity')
plt.savefig("linreg_linearity.pdf")


# %%
#Get energy resolution from distribution of predictions
def res(preds,energy):
    return norm.fit(preds)[1]/energy

energy_list = [150,100,50,20,10]
resolutions = res(y_preds_150GeV,150), res(y_preds_100GeV,100), res(y_preds_50GeV,50), res(y_preds_20GeV,20), res(y_preds_10GeV,10)


# %%
#Curve fit for energy resolution as a function of energy
def f(E, a, b):
    return np.sqrt((a/np.sqrt(E))**2 + b**2)

popt, pcov = curve_fit(f, energy_list, resolutions)

# %%
#Include gaussian fit in loss fn

popt, pcov

# %%
plt.plot(range(150),f(range(1,151),popt[0]))
plt.scatter(energy_list,resolutions)
plt.xlabel('Energy (GeV)')
plt.ylabel('Resolution')
plt.title('Energy Resolution')
plt.savefig("linreg_res.pdf")

# %%
torch.save(obj=model_0.state_dict, f="/home/dmisra/eic/model_0")
