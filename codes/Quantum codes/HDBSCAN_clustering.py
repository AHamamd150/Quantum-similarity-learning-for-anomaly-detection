import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras 
from keras import Model,layers
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Conv2D
import pennylane as qml 
from pennylane import numpy as np
import sklearn
tf.keras.backend.set_floatx('float32')
from scipy.ndimage import gaussian_filter
from keras.models import Model, Sequential
from keras.layers import Dense,LayerNormalization, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.constraints import UnitNorm
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "rm"
plt.rcParams['text.usetex'] = True
#########################
import qutip 
from qutip import Bloch,about, basis,  sesolve, sigmax,sigmay, sigmaz
from mpl_toolkits.mplot3d import Axes3D
###############################################
##Upload the projected events onto the qubit for the two clusters###
###############################################
test_data = np.loadtxt('test_points.txt')
x1 = np.loadtxt('expX0_1.txt');x2 = np.loadtxt('expX0_2.txt');x3 = np.loadtxt('expX0_3.txt');
y1 = np.loadtxt('expY0_1.txt');y2 = np.loadtxt('expY0_2.txt');y3 = np.loadtxt('expY0_3.txt');
z1 = np.loadtxt('expZ0_1.txt');z2 = np.loadtxt('expZ0_2.txt');z3 = np.loadtxt('expZ0_3.txt');
###############################################
import hdbscan
## Initialize the HDBSCAN ########
cluster_h = hdbscan.HDBSCAN(min_cluster_size=1000,min_samples=10, prediction_data=True  ,metric='euclidean');
## Fit the HDBSCAN to create the clusters####
cluster_hd=cluster_h.fit(data);
##### test how many clusters are clreated##
print(np.unique(cluster_hd.labels_))
#######################################
## define the new fidelity according how close the test point
## to each of the two clusters created by the HDBSCAN#
#######################################
from sklearn.neighbors import NearestNeighbors
#######################################
def fidelity(train_1,train_2,x,W):
    n1 = NearestNeighbors(n_neighbors=100)
    n2 = NearestNeighbors(n_neighbors=100)
    n1.fit(train_1)
    n2.fit(train_2)
    d1,d2,d3 = qubit_embed(x,W)
    data = np.vstack((d1,d2,d3)).T
    fid = np.array(n1.kneighbors(data))[0,:,0] - np.array(n2.kneighbors(data))[0,:,0]
    return fid
    
######################################
### Extract the training points that belong to each
## of the two clusters 
############################################
data1 = data[cluster_hd.labels_==0] ## data in the first cluster
data2 = data[cluster_hd.labels_==1] ## data in the second cluster
#############################################
## Load the trained network##
######################
model_HS = tf.keras.models.load_model('model_HS.keras')
# first get the weights of the trained network
W = model_HS.layers[-1].get_weights()
W_new = np.array(W).reshape(num_layers,round(qubits/2),3)
# New quantum circuit to measure the data embedding
def embedding_qlayer1(x):
    batched = qml.math.ndim(x) > 1
    x = qml.math.T(x) if batched else x
    qml.Rot(x[0],x[1],x[2], wires=0)
    qml.Rot(x[3],x[4],x[5], wires=1)
    qml.Rot(x[6],x[7],x[8], wires=2)

dev2 = qml.device('default.qubit.tf', wires=int(qubits/2),shots=10)
@qml.qnode(dev2, interface="tf")
def qcircuit1(params,inputs):
    for i in range(num_layers): 
        embedding_qlayer1(inputs)
        qml.StronglyEntanglingLayers(weights=tf.expand_dims(params[i,:,:],0), wires=range(int(qubits/2)), imprimitive=qml.ops.CZ)
    return qml.expval(qml.PauliX(0)),qml.expval(qml.PauliY(0)),qml.expval(qml.PauliZ(0)) 
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
def qubit_embed(x_in,weights):
  '''Argus:
  x_in: 1d array test images

  weights: saved weights of the quanutm layer

  Output: 1d array of length 3 with expectation values measured on the X,Y and Z axes  
  '''
  x,y,z=[],[],[]
  for i in tqdm(range(len(x_in)),ascii=True):
    h = embedding_network(x_in[i].reshape(-1,n_constit,n_channels))
    hq= qcircuit1(weights,h)
    x.append(hq[0])
    y.append(hq[1])
    z.append(hq[2])
  return x,y,z  
############################
fpr_ = fidelity(data1,data2,test_data,W_new)
tpr_ = fidelity(data1,data2,test_data,W_new)    












