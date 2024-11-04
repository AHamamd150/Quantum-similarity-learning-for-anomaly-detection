import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore')
import sys
import pennylane as qml 
from pennylane import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import layers
from tensorflow.keras.optimizers import Adam
from pennylane import numpy as np
import random    
import sklearn
tf.keras.backend.set_floatx('float32')
import keras.backend as K
from keras.datasets import mnist,cifar10,fashion_mnist
from tqdm import tqdm
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten,  Conv2DTranspose, MaxPooling2D, UpSampling2D
initializer = tf.keras.initializers.HeNormal(seed=42)
from keras.constraints import UnitNorm
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score,auc,RocCurveDisplay,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
#####################################
## Define the hyper parameters of the model ######
##################################### 
k_event = 100_000
test_ratio = 0.1 
attention_mask =False
num_classes=2  
batch_size= 128  
epoch = 20     
mlp_units = [256,128]    
dropout_rate = 0.2 
lr = 0.0001 
num_heads = 8  
hidden_dim= 60    
num_transformers= 2
n_constit = 11    
n_channels = 10     
input_shape = (n_constit,n_channels)
mlp_head_units = [64,n_channels]
#############################################
### Adjust the training and test data sets ################
#############################################
k_event=10_000 
sig_file = 'input_ML/Transformer_signal_QML.npz'
bkg1_file = 'input_ML/Transformer_bkgjets_QML.npz'
bkg2_file = 'input_ML/Transformer_bkgzzz_QML.npz'
signal=np.load(sig_file,allow_pickle=True)['arr_0']
signal = shuffle(signal)[:k_event]
bkg1 =np.load(bkg1_file,allow_pickle=True)['arr_0']
bkg2 =np.load(bkg2_file,allow_pickle=True)['arr_0']
bkg_file = np.concatenate((bkg1,bkg2))
bkg_file = shuffle(bkg_file)
background = bkg_file[:k_event]
print('###============================================###')
print(f'''Shape of the signal= {signal.shape}
Shape of the background= {background.shape}''')
print('###============================================###')
x1_data = np.concatenate((signal, background))
y1_data = np.array([1]*len(signal)+[0]*len(background))
x_data,y_data= sklearn.utils.shuffle(x1_data, y1_data) # shuffle both 
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,shuffle=True, test_size=test_ratio)
x_train= np.array(X_train).astype('float32')
x_test= np.array(X_test).astype('float32')
print('###============================================###')
print(f'''Shape of the training set= {x_train.shape}
Shape of the test set= {x_test.shape}''')
print('###============================================###')
###################################################
###################################################
###################################################
### Define the augmetation functions #########################
###################################################
def augment(x):
  ## Order of the feature inputs
  ## I1,I2,I3,I4,pt,eta,phi,M,charge # 
  # Smear the transverse momentum
  noise_4 = tf.random.normal(mean=x[:,:,4], shape=x[:,:,4].shape)
  # Smear the eta and phi 
  noise_5 = tf.random.normal(mean=x[:,:,5],stddev=1/x[:,:,4], shape=x[:,:,5].shape)
  noise_6 = tf.random.normal(mean=x[:,:,6],stddev=1/x[:,:,4], shape=x[:,:,6].shape)
  #### Rotate the phi component######
  noise_6_new =  x[:,:,6]*tf.math.cos(tf.random.uniform(shape=x[:,:,6].shape,minval=-3, maxval=3))
  # Modify the values using TensorFlow operations
  updated_col_4 =  noise_4
  updated_col_5 =  noise_5
  updated_col_6 =  noise_6_new
                                    
  # Create a new tensor with the updated columns
  x = tf.concat([x[:,:,:4], tf.expand_dims(updated_col_4, axis=-1), tf.expand_dims(updated_col_5, axis=-1), tf.expand_dims(updated_col_6, axis=-1), x[:,:,7:]], axis=-1)
  return x
  
##################################
### Define the MLP for the transformer network#  
##################################
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
 
##################################
### Attention mask to allivate the zero padding #  
##################################    
def masked_fill(tensor, idx,value,n_constit):
    pad_mask = tf.math.not_equal(tensor[:,:,idx],0)
    ##mask value == 0 --> no attention
    ## mask value ==1 --> Compute the attention
    ## for reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
    mask = tf.where(pad_mask, tf.cast(tf.fill(tf.shape(tensor[...,-1]), value),tf.float32), tensor[:,:,idx])
    mask = tf.repeat(mask[:,tf.newaxis],n_constit,axis=1)
    mask=tf.einsum("...ij->...ji",mask) 
    return mask

######################################
## Padding function to adjust the size of the input data#
######################################

def padding(x,n,a):
    arr = np.empty(shape=[0,n,x.shape[-1]])
    for i in range(len(x)):
        sys.stdout.write('\r'+'%s %s/%s'%(str(a),str(i),str(len(x))))
        if x.shape[1] < n:
            arr= np.append(arr,np.expand_dims(np.concatenate((x[i,:,:],np.zeros((n-x.shape[1],x.shape[-1])))),0),axis=0)
        elif x.shape[1] > n:
            arr= np.append(arr,np.expand_dims(x[i,:n,:],0),axis=0)
        else:
            arr= np.append(arr,np.expand_dims(x[i,:,:],0),axis=0)
    return np.array(arr)
######################
## Transformer encoder Layer#
######################
class transformer_encoder_layer_MHSA:
    def __init__(self,heads, dropout,mask,mlp_units,hidden_dim):
        self.heads = heads
        self. dropout=dropout
        self.mask = mask
        self.mlp_units = mlp_units
        self.hidden_dim=hidden_dim

    
    def layer(self,x):     
        MHA = layers.MultiHeadAttention(num_heads=self.heads, key_dim=self.hidden_dim, dropout=self.dropout)    
        x1 = layers.LayerNormalization()(x)
        attention_output,weights = MHA(x1, x1, attention_mask = self.mask,return_attention_scores=True)
        
        x2 = layers.Add()([attention_output, x])
        x3 = layers.LayerNormalization()(x2)
        x3 = mlp(x3, hidden_units=self.mlp_units, dropout_rate=self.dropout)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        return encoded_patches
##############################################
#### Create the full encoder network with the projection head####
##############################################
def create_Part_classifier():
    inputs = layers.Input(shape=input_shape) 
    if attention_mask:
        attentionmask= masked_fill(inputs,2,1,inputs.shape[1])
    else:
        attentionmask= None
    transformer_encoder = transformer_encoder_layer_MHSA(num_heads,dropout_rate,attentionmask,mlp_head_units,hidden_dim)
    encoded_patches = layers.LayerNormalization()(inputs)    
    for _ in range(num_transformers):
        encoded_patches = transformer_encoder.layer(encoded_patches)
       
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    #representation = layers.Dropout(dropout_rate)(representation)
    # Add MLP.
    #features = mlp(representation, hidden_units=mlp_units, dropout_rate=dropout_rate)
    logits = layers.Dense(9,activation='linear')(representation)
    out_ = layers.Dense(1,activation='sigmoid')(logits)
    model = keras.Model(inputs=inputs, outputs=out_)
    return model
######################################################
####### Create the network and print out the summary################
######################################################
embedding_network = create_Part_classifier()
embedding_network.summary()
#######################################################


#####################
####
Model_HS = Model(embedding_network.input,embedding_network.layers[-2].output)

#################################
# define the number of qubits #############
# define the number of repeated quantum layers#
################################# 
qubits = 6
num_layers= 2
#######define the embedding quantum layer ####
def embedding_qlayer(x):
   #batched = qml.math.ndim(x) > 1
    #x = qml.math.T(x) if batched else x
    qml.Rot(x[0],x[1],x[2], wires=1)
    qml.Rot(x[3],x[4],x[5], wires=2)
    qml.Rot(x[6],x[7],x[8], wires=3)

    qml.Rot(x[9],x[10],x[11], wires=4)
    qml.Rot(x[12],x[13],x[14], wires=5)
    qml.Rot(x[15],x[16],x[17], wires=6)
#####################    
## Define the quantum circuit#
#####################
dev = qml.device('default.qubit.tf', wires=qubits+1)
@qml.qnode(dev)
def qcircuit(params,inputs):
    for i in range(num_layers): 
        #qml.AngleEmbedding(features=inputs, wires=range(1,qubits+1), rotation='Y')
        #qml.IQPEmbedding(features=inputs[:,:3], wires=[1,2,3])
        #qml.IQPEmbedding(features=inputs[:,3:], wires=[4,5,6])
        qml.StronglyEntanglingLayers(weights=np.expand_dims(params[i,:int(qubits/2),:],0), wires=[1,2,3], imprimitive=qml.ops.CZ)
        qml.StronglyEntanglingLayers(weights=np.expand_dims(params[i,int(qubits/2):,:],0), wires=[4,5,6], imprimitive=qml.ops.CZ)
        embedding_qlayer(inputs)
    qml.Hadamard(wires=0)
    for k in range(3):
        qml.CSWAP(wires=[0, k + 1, qubits/2 + k + 1])
    qml.Hadamard(wires=0)

    return qml.expval(qml.PauliZ(0)) 
    
#### get the sahpe of the quantum circuit weights ######################    
shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=qubits)

#### Initialize the weights randomaly ####
params = np.random.random(size=shape)
## create the quanutm layer
shapes = {"params": qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=int(qubits/2))}
qlayer = qml.qnn.KerasLayer(qcircuit, shapes, output_dim=1,name='Quantum_Layer')


##########################################
## below define the hybrid classical-quantum network ######
##########################################
input_1 = layers.Input(input_shape)
input_2 = layers.Input(input_shape)

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

output1 = tf.concat([tower_1,tower_2],1)
### Incorporate the quantum layer into the classical network
output2 = qlayer(output1)

model_HS = keras.Model(inputs=[input_1, input_2], outputs=output2)
model_HS.summary()
#########
## define how to batch the training data ##

def iterate_minibatches(inputs, targets, batch_size):
    """
    A generator for batches of the input data

    Args:
        inputs (array[float]): input data
        targets (array[float]): targets

    Returns:
        inputs (array[float]): one batch of input data of length `batch_size`
        targets (array[float]): one batch of targets of length `batch_size`
    """
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]
####################################
## Define the quantum loss function ###########
####################################
def overlaps(params,out1,out2):
    swap = 0
    for i in range(len(out1)): 
        output2 = np.concatenate([out1[i],out2[i]],0)
        output_2 = qcircuit(params,output2)
        swap += output_2 
    return swap
    
def cost(weights, A, B):
    aa = overlaps(weights,A, augment(A))
    ab = overlaps(weights,A,B)
    d_hs = -ab + aa 
    return 1 - d_hs    
    
def make_pairs(x, y):
    pairs = []
    for idx1,idx2 in zip(x,y): 
        pairs += [[idx1, idx2]]
    return np.array(pairs)
####################################################################
from pennylane.optimize import AdamOptimizer 
opt = AdamOptimizer()
##########################                
### Train the quantum network ####
##########################
for it in range(epoch):
  print(f'Run Epoch number: {it+1}')
  q = 0
  for xbatch1,xbatch2 in iterate_minibatches(pairs_train[:1000,0], pairs_train[:1000,1], batch_size=batch_size):
    q+=1
    xbatch1_ = np.array(Model_HS(xbatch1))
    xbatch2_ = np.array(Model_HS(xbatch2))
    params= opt.step(cost,params,xbatch1_,xbatch2_)
    print(cost(params,xbatch1_,xbatch2_))
    sys.stdout.write('\r'+ 'Processing batch No: %s / %s '%(q,len(xbatch1)))
   
    
##########################################

##########################################
np.savetxt('l_1.txt',l1)
np.savetxt('l_2.txt',l2)
#######################################
## After traininf the network, project the points onto the#
## Bloch sphere  of the read out qubit #############
########################################

# first save the weights for the quantum layer 
W = model_HS.layers[-1].get_weights()
W_new = np.array(W).reshape(num_layers,round(qubits/2),3)
W_rand_new = np.array(W_rand[0]).reshape(num_layers,round(qubits/2),3)
###########################
# New quantum circuit to measure the data embedding
def embedding_qlayer1(x):
    batched = qml.math.ndim(x) > 1
    x = qml.math.T(x) if batched else x
    qml.Rot(x[0],x[1],x[2], wires=0)
    qml.Rot(x[3],x[4],x[5], wires=1)
    qml.Rot(x[6],x[7],x[8], wires=2)

############## Large number of shots - 10000 ##################
dev2 = qml.device('default.qubit.tf', wires=int(qubits/2),shots=10000)
@qml.qnode(dev2, interface="tf")
def qcircuit2(params,inputs):
    for i in range(num_layers): 
        embedding_qlayer1(inputs)
        qml.StronglyEntanglingLayers(weights=tf.expand_dims(params[i,:,:],0), wires=range(int(qubits/2)), imprimitive=qml.ops.CZ)
    return qml.expval(qml.PauliX(0)),qml.expval(qml.PauliY(0)),qml.expval(qml.PauliZ(0)) 

############## Small number of shots - 10 shots ######################
dev3 = qml.device('default.qubit.tf', wires=int(qubits/2),shots=10)
@qml.qnode(dev3, interface="tf")
def qcircuit3(params,inputs):

    for i in range(num_layers): 
        embedding_qlayer1(inputs)
        qml.StronglyEntanglingLayers(weights=tf.expand_dims(params[i,:,:],0), wires=range(int(qubits/2)), imprimitive=qml.ops.CZ)

    return qml.expval(qml.PauliX(0)),qml.expval(qml.PauliY(0)),qml.expval(qml.PauliZ(0)) 
############################################################
from tensorflow.python.ops.numpy_ops import np_config 
np_config.enable_numpy_behavior()
#############################################################
def qubit_embed(x_in,weights,qcircuit):
  '''Argus:
  x_in: 1d array test images

  weights: saved weights of the quanutm layer

  Output: 1d array of length 3 with expectation values measured on the X,Y and Z axes  
  '''
  x,y,z=[],[],[]
  for i in tqdm(range(len(x_in)),ascii=True):
    h = embedding_network(x_in[i].reshape(-1,n_constit,n_channels))
    hq= qcircuit(weights,h)
    x.append(hq[0])
    y.append(hq[1])
    z.append(hq[2])
  return x,y,z  
##########################
expX0_1,expY0_1,expZ0_1 = qubit_embed(x_train_new[:10000,1],W_new,qcircuit2)
expX1_1,expY1_1,expZ1_1 = qubit_embed(x_train_new[:10000,0],W_new,qcircuit2)

expX0_2,expY0_2,expZ0_2 = qubit_embed(x_train_new[:10000,1],W_new,qcircuit3)
expX1_2,expY1_2,expZ1_2 = qubit_embed(x_train_new[:10000,0],W_new,qcircuit3)


expX0_0,expY0_0,expZ0_0 = qubit_embed(x_train_new[:10000,1],W_rand_new,qcircuit2)
expX1_0,expY1_0,expZ1_0 = qubit_embed(x_train_new[:10000,0],W_rand_new,qcircuit2)


np.savetxt('expX0_0.txt',expX0_0)
np.savetxt('expY0_0.txt',expY0_0)
np.savetxt('expZ0_0.txt',expZ0_0)
np.savetxt('expX0_1.txt',expX0_1)
np.savetxt('expY0_1.txt',expY0_1)
np.savetxt('expZ0_1.txt',expZ0_1)
np.savetxt('expX0_2.txt',expX0_2)
np.savetxt('expY0_2.txt',expY0_2)
np.savetxt('expZ0_2.txt',expZ0_2)

np.savetxt('expX1_0.txt',expX1_0)
np.savetxt('expY1_0.txt',expY1_0)
np.savetxt('expZ1_0.txt',expZ1_0)
np.savetxt('expX1_1.txt',expX1_1)
np.savetxt('expY1_1.txt',expY1_1)
np.savetxt('expZ1_1.txt',expZ1_1)
np.savetxt('expX1_2.txt',expX1_2)
np.savetxt('expY1_2.txt',expY1_2)
np.savetxt('expZ1_2.txt',expZ1_2)
############################
import qutip 
from qutip import Bloch,about, basis,  sesolve, sigmax,sigmay, sigmaz
from mpl_toolkits.mplot3d import Axes3D
import itertools
########################################
fig = plt.figure(figsize=(17,15),constrained_layout=True)##
########################################
## Plot the Bloch sphere using QuTip package# 
################################
ax1 = fig.add_subplot(1, 3,1,  projection='3d')
b1= qutip.Bloch(fig=fig, axes=ax1)
b1.clear()
b1.font_color='k'
b1.frame_alpha=0.15
b1.sphere_alpha=0.2
b1.font_size=15
b1.sphere_color='darkblue'
b1.frame_color='k'
b1.point_size = [10,10]
b1.point_marker=['o','o'] 
b1.xlabel=['', '']
b1.ylabel=['', '']
b1.add_points([np.array(expX0_0).flatten(),np.array(expY0_0).flatten(),np.array(expZ0_0).flatten()])
b1.add_points([np.array(expX1_0).flatten(),np.array(expY1_0).flatten(),np.array(expZ1_0).flatten()])
#b1.point_color=['tab:blue','tab:orange']
#b1.view=[-10,10]
b1.zlpos = [1.2, -1.35]
b1.render()
ax1.set_title('10000 shots', fontsize=15,pad=20);
##########################
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
b2= qutip.Bloch(fig=fig, axes=ax2)
b2.clear()
b2.font_color='k'
b2.frame_alpha=0.15
b2.sphere_alpha=0.2
b2.font_size=15
b2.sphere_color='darkblue'
b2.frame_color='k'
b2.point_size = [10,10]
b2.point_marker=['o','o'] 
b2.xlabel=['', '']
b2.ylabel=['', '']
b2.add_points([np.array(expX0_1).flatten(),np.array(expY0_1).flatten(),np.array(expZ0_1).flatten()])
b2.add_points([np.array(expX1_1).flatten(),np.array(expY1_1).flatten(),np.array(expZ1_1).flatten()])
#b1.point_color=['tab:blue','tab:orange']
#b1.view=[-10,10]
b2.zlpos = [1.2, -1.35]
b2.render()
ax2.set_title('10000 shots', fontsize=15,pad=20);
##########
###################
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
b3= qutip.Bloch(fig=fig, axes=ax3)
b3.clear()
b3.font_color='k'
b3.frame_alpha=0.15
b3.sphere_alpha=0.2
b3.font_size=15
b3.sphere_color='darkblue'
b3.frame_color='k'
b3.point_size = [10,10]
b3.point_marker=['o','o'] 
b3.xlabel=['', '']
b3.ylabel=['', '']
b3.add_points([np.array(expX0_2).flatten()/1.2,np.array(expY0_2).flatten()/1.2,np.array(expZ0_2).flatten()/1.2])
b3.add_points([np.array(expX1_2).flatten()/1.32,np.array(expY1_2).flatten()/1.32,np.array(expZ1_2).flatten()/1.32])
#b2.point_color=['tab:blue','tab:orange']
#b2.view=[-10,10]
b3.zlpos = [1.2, -1.35]
b3.render()
ax3.set_title('10 shots', fontsize=15,pad=20);
plt.show()
######################################



