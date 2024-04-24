#EVALUACIÓN DE NUEVOS DATOS

#Sección: Importar datos de Excel
import pandas as pd

#ubicación del documento de excel con los datos
path = "C:/Users/marti/OneDrive/Escritorio/new_data_dkms.xlsx"
#Nombre de la columna con los códigos SMILES
smiles = "SMILES"
#Nombre de la columna con los datos experimentales
exp = "experimental_data"

datos = pd.read_excel(path)
smiles_list = datos[smiles].to_list()
exp_data = datos[exp].to_list()

print(smiles_list)
print(exp_data)

#Sección: Crear un objeto mol por cada código smiles y guardarlos en una lista
from rdkit import Chem

mol_list = [Chem.MolFromSmiles(m) for m in smiles_list]

print(mol_list[0])

#Sección: Inicializa los featurizers, los cuales normalizan los datos de entrada.
#Con ellos se determina el número de nodos n_feats (características de los átomos) y el número de edges
#(características de los enlaces) e_feats, para la representación gráfica de las moléculas.

from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph

atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()

e_feats = bond_featurizer.feat_size('e')
n_feats = atom_featurizer.feat_size('h')

#Sección: Transforma los objetos mol en gráficas usando los featurizers,
#y guarda las gráficas en una lista llamada X.

X = [mol_to_bigraph(m, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for m in mol_list]

print(X[0])

#Sección: Transformar la lista de datos experimentales en un arreglo de numpy de 1 columna,
#estandarizarlos y convertirlos en tensores.

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

exp_data = np.array(exp_data)
exp_data = exp_data.reshape(-1, 1)
y_scaler = StandardScaler()
y_scaled = torch.Tensor(y_scaler.fit_transform(exp_data))

#Sección: Crear una lista de tuples(Gráfica que representa a la molécula,
#Datos experimentales estandarizados en forma de tensores).
# Estos son los datos de entrada que alimentan a la red neuronal.

input_data = list(zip(X, y_scaled))

#Sección: Crear un modelo con las mismas características que el modelo entrenado

from dgllife.model.model_zoo import GATPredictor

model = GATPredictor(in_feats=n_feats)

model

#Sección: Cargar el modelo guardado en el nuevo objeto "model"

model.load_state_dict(torch.load('./model_saved.pth'))

#Sección: Evaluar nuevos datos, almacenar las predicciones en un arreglo, y los datos reales en otro.

# Evaluate

import dgl
from torch.utils.data import DataLoader

device = 'cpu'

def collate(sample):
    graphs, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)


test_loader = DataLoader(input_data, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)

model.eval()
preds = []
labs = []
for i, (bg, labels) in enumerate(test_loader):
    labels = labels.to(device)
    atom_feats = bg.ndata.pop('h').to(device)
    bond_feats = bg.edata.pop('e').to(device)
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

#model.eval()


