# main.py
from datetime import datetime
import deepchem as dc
import numpy as np 
import csv
from deepchem.feat.mol_graphs import ConvMol
import pandas as pd
from mygraphconvmodel import MyGraphConvModel
from deepchem.models import KerasModel

def read_data(input_file_path, output_file_path):
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(tasks=prediction_tasks, feature_field="SMILES", featurizer=featurizer)
    dataset = loader.featurize(input_file_path, shard_size=8192)
    
    trainarray=[]
    smiles = dataset.ids
    values = dataset.y    
    
    for i in range(dataset.ids.size):
      entry=[str(smiles[i]),str(values[i][0])]
      trainarray.append(entry)

    with open(output_dir+output_file_path,mode='w') as csv_file:
        wr=csv.writer(csv_file,quoting=csv.QUOTE_NONE)
        wr.writerows(trainarray)

    return dataset

def make_predictions(model,dataset,batch_size, output_dir, result_dir, read_out_string ):
  predictions = model.predict_on_generator(data_generator(dataset,batch_size=batch_size))
  predictions = reshape_y_pred(dataset.y, predictions)
  write_predictions(predictions, 'Auslesen'+read_out_string+'Prediction.csv')
  datasets(output_dir+'Auslesen'+read_out_string+'.csv',output_dir+'Auslesen'+read_out_string+'Prediction.csv',result_dir)
 
def data_generator(dataset,batch_size,epochs=1,modelX = None):
  for epoch in range(epochs):
    if modelX is not None:
      print("start epoch "+ str(epoch+1))
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(batch_size,deterministic=True, pad_batches=True)):
      multiConvMol = ConvMol.agglomerate_mols(X_b)
      inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership)]
      for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
      labels = [y_b]
      weights = [w_b]
      yield (inputs, labels, weights)
    if modelX is not None:
      print("end epoch "+ str(epoch+1))
    if modelX is not None:
      print("#####Making step predictions.")
      modelX.model_dir = basis+"StepModelle/"
      modelX.fit_generator(data_generator(train_dataset,batch_size,epochs=1))
      modelX.save_checkpoint()
      print("#####Making ModelX TrainSet predictions.")
      make_predictions(model=modelX,dataset= train_dataset,batch_size=batch_size, output_dir=output_dir, result_dir=result_dir, read_out_string='Trainset')
      print("#####Making ModelX Valset predictions.")
      make_predictions(model=modelX,dataset= test_dataset,batch_size=batch_size, output_dir=output_dir, result_dir=result_dir, read_out_string='Valset')

def reshape_y_pred(y_true, y_pred):
 
  n_samples = len(y_true)
  retval = np.vstack(y_pred)
  return retval[:n_samples]

def write_predictions(dataset, file_name):
  with open(output_dir+file_name,mode='w') as csv_file:
    wr=csv.writer(csv_file,quoting=csv.QUOTE_ALL)
    wr.writerows(dataset)

#Constants
prediction_tasks = ['S']
basis='./'
model_dir = basis


#model parameters
batch_size = 50
num_epochs = 130
neuronslayer1=64
neuronslayer2=128
dropout=0.1
learning_rate=0.0005

#training model
train= MyGraphConvModel(batch_size=batch_size,neuronslayer1=neuronslayer1,neuronslayer2=neuronslayer2,dropout=dropout)
keras_model = KerasModel(train, loss=dc.models.losses.L1Loss() ,learning_rate=learning_rate,  model_dir=model_dir)
print('---start training ---')
#if no step predictions needed, modelX => None
loss = keras_model.fit_generator(data_generator(train_dataset,batch_size=batch_size,epochs=num_epochs)) 
print('TRAINED MODEL Loss: %f' % loss)

#loading model
model2 = KerasModel(MyGraphConvModel(batch_size=batch_size,neuronslayer1=neuronslayer1,neuronslayer2=neuronslayer2,dropout=dropout), loss=dc.models.losses.L1Loss(), model_dir=model_dir)
model2.restore()
print('---start training predictions---')
#training predictions
make_predictions(model=model2,dataset= train_dataset,batch_size=batch_size)

#test predictions
print('---start test predictions---')
make_predictions(model=model2,dataset= test_dataset,batch_size=batch_size)

