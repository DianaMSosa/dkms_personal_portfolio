import dgl
import pandas as pd
import numpy as np
import torch

from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from dgllife.model.model_zoo import GATPredictor
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# IMPORTAR DATOS

# Ubicación del documento de excel con los datos
path = "C:/Users/marti/OneDrive/Escritorio/new_data_dkms.xlsx"
# Nombre de la columna con los códigos SMILES
smiles = "SMILES"
# Nombre de la columna con los datos experimentales
exp = "experimental_data"

datos = pd.read_excel(path)
smiles_list = datos[smiles].to_list()
exp_data = datos[exp].to_list()

# TRATAMIENTO DE DATOS PREVIO A LA EVALUACIÓN

mol_list = [Chem.MolFromSmiles(m) for m in smiles_list]

atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()
# e_feats = bond_featurizer.feat_size('e')
n_feats = atom_featurizer.feat_size('h')

X = [mol_to_bigraph(m, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for m in mol_list]

exp_data = np.array(exp_data)
exp_data = exp_data.reshape(-1, 1)
y_scaler = StandardScaler()
y_transformed = torch.Tensor(y_scaler.fit_transform(exp_data))

input_data = list(zip(X, y_transformed))

# IMPORTAR EL MODELO PREVIAMENTE ENTRENADO

model = GATPredictor(in_feats=n_feats)
model.load_state_dict(torch.load('./model_saved.pth'))

# EVALUACIÓN DE LOS NUEVOS DATOS

if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('Using CPU')
    device = 'cpu'
def collate(sample):
    graphs, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)

loader = DataLoader(input_data, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)

model.eval()
preds = []
labs = []
for i, (bg, labels) in enumerate(loader):
    labels = labels.to(device)
    atom_feats = bg.ndata.pop('h').to(device)
    # bond_feats = bg.edata.pop('e').to(device)
    atom_feats, labels = atom_feats.to(device), labels.to(device)
    y_pred = model(bg, atom_feats)
    labels = labels.unsqueeze(dim=1)

    # Inverse transform to get RMSE
    labels = y_scaler.inverse_transform(labels.reshape(-1, 1))
    y_pred = y_scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))

    preds.append(y_pred)
    labs.append(labels)

labs = np.concatenate(labs, axis=None)
preds = np.concatenate(preds, axis=None)

print("Correct values are:")
print(labs)
print("Predicted values are:")
print(preds)
