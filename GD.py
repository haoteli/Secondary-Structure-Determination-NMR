import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import pandas as pd
import os
import random
import pandas as pd
import sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'




data_ca = pd.read_csv(sys.argv[1])

#path = sys.argv[-1])

print('The csv file has the following column names:')
for i,j in enumerate(data_ca.columns):
  print('---------------------------------')
  print('Column' + str(i) + ':',j)

print('\n')

Manual_Sequence_Input = 'Yes'
PPM_axis_name = sys.argv[2]
Intensity_axis_name = sys.argv[3]
FASTA_Sequence = sys.argv[4]

min_ppm =  10
max_ppm =  195

Exclude_Aromatic_Side_Chain = "No" 


Re_reference =  0
Mode_vs_Mean = "Mode Shift"





if(Manual_Sequence_Input == 'Yes'):
    sequence = np.array([s.upper() for s in FASTA_Sequence],dtype = str)
else:
    try:
        columns = data_ca.columns
        column_name = columns[Seq_Use_Column]
        start = ''

        for i in data_ca[column_name].values:
            try:
                if(i.isalpha()):
                    start += str(i)
            except:
                break

    except:
        print('Error Occured. Check Sequence Column Number.')


    sequence = start
    print('Selected Sequence From Column:', sequence)
    sequence = np.array([s.upper() for s in sequence],dtype = str)




ppm = data_ca[PPM_axis_name].values - Re_reference
real_spectrum = data_ca[Intensity_axis_name].values

index1 = (ppm > min_ppm) 
index2 = (ppm < max_ppm)
index = index1*index2 # Indexing the range of data to minimize noise


real_spectrum = real_spectrum[index]
ppm = ppm[index] 

negative_inds = (real_spectrum < 0)
if(np.sum(negative_inds) > 1):
  mean_negative = real_spectrum[negative_inds].mean()
  real_spectrum = real_spectrum - mean_negative # Correcting the negative part of the initial spectrum

plt.grid()
plt.plot(ppm,real_spectrum)
plt.xlabel('PPM')
plt.xlim(max_ppm,min_ppm)
plt.ylabel('Intensity (Normalized)')
plt.savefig('Initial_Spectrum.svg')


real_spectrum = real_spectrum/real_spectrum.sum() # Normalization

#initialization = torch.zeros((len(sequence), 3),dtype = torch.float32) # Specifying initial guess
#initialization[:,0] = 3
initialization = None # Do not use a specific guess of initial state, start with equal mixture



shift_table = pd.read_csv('Shift Table.txt',sep=',', header=None)

shift_table = shift_table.rename({0:'Res Name', 1:'Atom Type', 2:'Secondary Structure', 3:'Mode Shift',4:'Mean Shift', 5: 'STD', 6:'Min Shift',7:'Max Shift'},axis = 1)

side_chain_table = pd.read_csv('side_chain_shifts.csv')
side_chain_table = side_chain_table.rename({'Amino Acid':'Res Name','CN2':'CH2'},axis = 1)
new_index = side_chain_table['Res Name'] # Use residue name as index 
side_chain_table = side_chain_table.set_index(new_index)

all_table_ss = shift_table['Secondary Structure'].values
all_table_ss[all_table_ss == 'E'] = 'B' # Replacing the notation of beta-sheet from E to B
shift_table['Secondary Structure'] = all_table_ss

atom_types = ['CA','CB','C']
SS = np.array(['C', 'H', 'B'])
residue_types = np.unique(shift_table['Res Name'].values)

all_possible_shifts = np.nan*np.zeros((len(residue_types),len(atom_types),len(SS))) # This is the vector that contain all possible shifts [Res type, atom type, SS type]

for i in range(len(residue_types)):
  entry = shift_table[shift_table['Res Name'] == residue_types[i]]
  for j in range(len(atom_types)):
    sub_entry = entry[entry['Atom Type'] == atom_types[j]]
    

    if(len(sub_entry) >= 3): # If has at least alpha, beta and coil three types
      for k in range(len(SS)):
        val = sub_entry[sub_entry['Secondary Structure'] == SS[k]][Mode_vs_Mean].values
        all_possible_shifts[i,j,k] = val
  

data_shift_table = np.zeros((len(sequence),len(atom_types),len(SS))) # Filling up the data table (CA, CB, C)
ind = 0
for res in sequence:
  data_shift_table[ind] = all_possible_shifts[np.argwhere(residue_types == res)]
  
  ind += 1

# Here we are running a separate list that is going to be initialized as te background of the spectra
backbone_means = []
for residue in sequence:
  residue_side_chain_table = side_chain_table.loc[residue].values[1:].astype(float)
  has_side_atom = ~np.isnan(residue_side_chain_table)
  for shift in residue_side_chain_table[has_side_atom]:
    backbone_means.append(shift)
backbone_means = np.array(backbone_means) # Side-chain shifts that will be used as background




Epochs =  100
print('Number of Epochs:', Epochs)
Learning_rate =  0.1
print('Learning Rate:', Learning_rate)
Freedom_ppm_back =  0
Freedom_ppm_side =  0

#GD_re_reference = "No" 

Fitting_Functional = "Lorentzian"
print('Fitting Functional:',Fitting_Functional)
Penalize_SS = "None" #@param ["None", "C", "B", "H"]
SS_Penalization_Effect =  0.1
Use_Initilization = "No" #@param ["Yes", "No"]
Initial_PB = 0.01
Initial_PH =  0.9

Plot_Name = "Default" 

if(Use_Initilization == 'Yes'):
  Initial_PC = 1 - Initial_PB - Initial_PH
  WB = np.log(Initial_PB)
  WH = np.log(Initial_PH)
  WC = np.log(Initial_PC)
  initialization = torch.zeros((len(sequence), 3),dtype = torch.float32) # Specifying initial guess
  initialization[:,np.argmax(SS == 'C')] = WC
  initialization[:,np.argmax(SS == 'B')] = WB
  initialization[:,np.argmax(SS == 'H')] = WH
else:
  initialization = None



GD_reference = True if (sys.argv[5] == 'True') else False





# Below is the class for Gradient Descent implementation
class GD(nn.Module):
  def __init__(self, protein_shift_table, N_backbone,initialization = None, use_re_reference = False): 
    super().__init__()
    # The initialization part defines all paramters

    N_res,N_atom_type,N_ss = protein_shift_table.shape
    self.shift_table = torch.from_numpy(protein_shift_table).float().to(device)
    if(initialization is not None):
      self.a = torch.nn.Parameter(initialization, requires_grad = True)
    else:
      self.a = torch.nn.Parameter(torch.ones((N_res, N_ss),dtype = torch.float32), requires_grad = True) # This is an even mix of states

    self.offset = torch.nn.Parameter(torch.zeros_like(self.shift_table,dtype = torch.float32), requires_grad = True) # Offset for alpha, beta, and carbonyl
    self.side_offset = torch.nn.Parameter(torch.zeros(N_backbone,dtype = torch.float32), requires_grad = True) # Offset for side-chains

    self.soft_a = torch.softmax(self.a, dim = 1)# Perform the softmax along the second dimension
    self.re_reference = torch.nn.Parameter(torch.zeros(1,dtype = torch.float32), requires_grad = use_re_reference)
    self.ita = torch.nn.Parameter(torch.zeros((1,2),dtype = torch.float32),requires_grad = True)

    self.b = torch.nn.Parameter(0.2*torch.ones((1,1),dtype = torch.float32), requires_grad = True)
    self.back_intensity_scaling = torch.nn.Parameter(torch.ones(N_res,dtype = torch.float32), requires_grad = True)
    #self.back_b = torch.nn.Parameter(0.2*torch.ones((N_res,N_atom_type),dtype = torch.float32), requires_grad = True)
  if(Fitting_Functional == 'Gaussian'):
    def g(self,x,mu,sig): # Gaussian density
      return 1./(np.sqrt(2.*np.pi)*sig)*torch.exp(-torch.pow((x - mu)/sig, 2.)/2)
  elif(Fitting_Functional == 'Lorentzian'):
    def g(self,x,mu,sig): # Lorenztian density
      return( (1/np.pi)*(0.5*sig)/((x-mu)**2  + (0.5*sig)**2) )
  elif(Fitting_Functional == 'Pseudo-Voigt (Linear Combination)'):
    def Gaussian(self,x,mu,sig): # Gaussian density
      return 1./(np.sqrt(2.*np.pi)*sig)*torch.exp(-torch.pow((x - mu)/sig, 2.)/2)

    def Lorentzian(self,x,mu,sig): # Lorenztian density
      return( (1/np.pi)*(0.5*sig)/((x-mu)**2  + (0.5*sig)**2) )

    def g(self,x,mu,sig): # Lorenztian density
      self.soft_ita = torch.softmax(self.ita, dim = 1)
      return( (self.soft_ita[0,0]*self.Gaussian(x,mu,sig) + self.soft_ita[0,1]*self.Lorentzian(x,mu,sig)) )

  def three_in_one(self, ppm):
    # This function sums the contribution of individual residues together
    summed_spectrum = 0
    self.distribution_loss = 0
    for atom_type_i in range(self.shift_table.shape[1]): # For each atom type
      for ss_type_j in range(self.shift_table.shape[2]): # For each secondary structure
        for residue_k in range(self.shift_table.shape[0]): # For each residue in the sequence
          if(~torch.isnan(self.shift_table[residue_k,atom_type_i,ss_type_j])):
            
           #summed_spectrum += self.back_intensity_scaling2[residue_k]*self.g(ppm, self.shift_table[residue_k,atom_type_i,ss_type_j] + self.offset[residue_k,atom_type_i,ss_type_j] - self.re_reference, self.b2.squeeze())*self.soft_a[residue_k,ss_type_j]
           summed_spectrum += self.g(ppm, self.shift_table[residue_k,atom_type_i,ss_type_j] + self.offset[residue_k,atom_type_i,ss_type_j] - self.re_reference, self.b2.squeeze())*self.soft_a[residue_k,ss_type_j]
           
            
    
    return(summed_spectrum)

  def forward(self, ppm,background_means = None, costum_state = None):
    # When called forward, it calculates the itensity at the location where the shift is known
    ppm = torch.from_numpy(ppm).float().to(device)
    if(costum_state is None):
      self.soft_a = torch.softmax(self.a, dim = 1)# Perform the softmax along the second dimension so that they sum to 1
    elif(costum_state is not None):
      self.soft_a = torch.from_numpy(costum_state)
    else:
      print('soft_a not defined!')
      
    
    self.b2 = self.b**2 # This is the procedure to make sure the std is always positive
    self.b2 = torch.max(self.b2, torch.tensor([0.001]).to(device)) # Any gaussian with 0 width is not allowed
    self.back_intensity_scaling2 = self.back_intensity_scaling**2
    if(self.b2 == torch.tensor([0.001]).to(device)):
      print('b2 reached minimum, consider changing parameters')


    self.b2 = self.b2.to(device)
    #self.back_b2 = (self.back_b**2).to(device)
    
    sim_spctra = self.three_in_one(ppm)

    if background_means is not None: # If account for side-chain Carbons
      
      background = 0
      for i in range(len(background_means)):
        background += self.g(ppm, background_means[i] + self.side_offset.squeeze()[i] - self.re_reference, self.b2.squeeze())

      sim_spctra += background


    sim_spctra = sim_spctra/torch.sum(sim_spctra)
    
    return(sim_spctra)






Epochs = Epochs

pred_percent_C = []
pred_percent_H = []
pred_percent_B = []

loss_collect = []


model = GD(data_shift_table,N_backbone = len(backbone_means),initialization=initialization, use_re_reference = GD_reference).to(device) # The model that will optimize the given spectrum


optimizer = torch.optim.Adam(model.parameters(), lr = Learning_rate)



for epoch in range(Epochs): # Training model
  sim_spectrum = model(ppm,background_means = backbone_means)
  offset_ind = (torch.abs(model.offset) > Freedom_ppm_back)
  side_offset_ind = (torch.abs(model.side_offset) > Freedom_ppm_side)
  
  loss_offset = ((torch.abs(model.offset[offset_ind]) - Freedom_ppm_back)**2).sum()  + ((torch.abs(model.side_offset[side_offset_ind]) - Freedom_ppm_side)**2).sum()
  
  if(Penalize_SS != 'None'):
    loss_offset += (SS_Penalization_Effect*model.soft_a[:,np.argmax(SS == Penalize_SS)].sum()/len(model.soft_a))**2
  
  loss = loss_offset + torch.log(torch.mean(torch.abs(sim_spectrum - torch.from_numpy(real_spectrum/real_spectrum.sum()).to(device))**2)) 
  loss_collect.append(loss.item())
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

pred_percent_C.append(model.soft_a[:,0].detach().cpu().numpy().sum()/len(model.soft_a))
pred_percent_H.append(model.soft_a[:,1].detach().cpu().numpy().sum()/len(model.soft_a))
pred_percent_B.append(model.soft_a[:,2].detach().cpu().numpy().sum()/len(model.soft_a))


if(GD_reference):
  print('GD re-reference: ', model.re_reference.item())
if(Fitting_Functional == 'Pseudo-Voigt (Linear Combination)'):
  print('Gaussian-Lorentzian Characterisics:', model.soft_ita[0,0].item(), model.soft_ita[0,1].item())
# Plotting spectra
sim_spectrum = sim_spectrum.detach().cpu().numpy()
plt.figure(figsize = (20,10),dpi = 300)
plt.subplot(121)
plt.plot(ppm,real_spectrum/real_spectrum.sum(),label = 'Exp Spectrum',color = 'black')
plt.plot(ppm, sim_spectrum, label = 'Simulated Spectrum',color = '#EC008C')
plt.legend()
plt.xlim(190,10)
plt.grid()
plt.ylabel('Normalized Intensity')
plt.xlabel('Shift (ppm)')

# Plotting simulated distribution
distribution = [pred_percent_C[-1],pred_percent_H[-1],pred_percent_B[-1]]

plt.subplot(122)
colors = ['black', 'red', '#3A53A4']

_, _, autotexts = plt.pie(distribution, explode=[0.001,0,0], labels=SS, autopct='%1.1f%%',colors = colors);
for autotext in autotexts:
    autotext.set_color('white')
plt.title('Simulated Distribution');
#plt.show()
if(Plot_Name == 'Default'):
    filename = Intensity_axis_name + '_Result' + '.svg'
    print('Saved Figure:',filename)
else:
    filename = Plot_Name + '.svg'

plt.savefig(filename)


print('\n')

print('Coil:', np.round(distribution[0],2))
print('Helix:',np.round(distribution[1],2))
print('Sheet:',np.round(distribution[2],2))


print('\n')
# plt.plot(np.array(loss_collect))
# plt.grid()
# plt.xlabel('Epochs')
# plt.ylabel('Fitting Loss')

