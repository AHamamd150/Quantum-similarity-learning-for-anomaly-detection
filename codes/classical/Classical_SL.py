import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import sys
#import pennylane as qml 
#from pennylane import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
import random    
import sklearn
tf.keras.backend.set_floatx('float32')
import keras.backend as K
#from keras.datasets import mnist,cifar10,fashion_mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten,  Conv2DTranspose, MaxPooling2D, UpSampling2D
#initializer = tf.keras.initializers.HeNormal(seed=42)
#from keras.constraints import UnitNorm
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
#from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score,auc,RocCurveDisplay,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
if not tf.test.gpu_device_name():
   warnings.warn('No GPU found.....')
   sys.exit('No GPU found.....')
else:
   print('Default GPU device :{}'.format(tf.test.gpu_device_name()))
#tf.config.optimizer.set_jit(True)
#strategy = tf.distribute.MirroredStrategy()
###################################################################### 
test_ratio = 0.2 
attention_mask =False
num_classes=2  
batch_size= 128 
epoch = 20     
mlp_units = [256,128]  
dropout_rate = 0.2 
lr = 0.001
num_heads = 8
hidden_dim= 1000    
num_transformers= 2
n_constit = 11    
n_channels = 10     
input_shape = (n_constit,n_channels)
mlp_head_units = [128,n_channels]
#############################################
k_event=100_000 
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
        attentionmask= masked_fill(inputs,6,1,inputs.shape[1])
    else:
        attentionmask= None
    transformer_encoder = transformer_encoder_layer_MHSA(num_heads,dropout_rate,attentionmask,mlp_head_units,hidden_dim)
    encoded_patches = layers.LayerNormalization()(inputs)    
    for _ in range(num_transformers):
        encoded_patches = transformer_encoder.layer(encoded_patches)
       
    # Create a [batch_size, projection_dim] tensor.
    #representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(encoded_patches)
    #representation = layers.Dropout(dropout_rate)(representation)
    # Add MLP.
    #features = mlp(representation, hidden_units=mlp_units, dropout_rate=dropout_rate)
    #representation= layers.Normalization()(representation)
    logits = layers.Dense(500,activation='linear')(representation) #### Projection head
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
    
######################################################
####### Create the network and print out the summary################
######################################################
embedding_network = create_Part_classifier()
embedding_network.summary()
#######################################################
#### Create the contrastive loss function ##########################
#######################################################
def contrastive_loss(x_i,x_j, temperature=0.05):
    # Get the batch size
    batch_size = tf.shape(x_i)[0]
    # Normalize the embeddings (similar to F.normalize in PyTorch)
    zi = tf.math.l2_normalize(x_i, axis=1)
    zj = tf.math.l2_normalize(x_j, axis=1)
    ## Compute the similarity between all positive and negative pairs
    similarity = tf.matmul(zi,zj, transpose_b=True)
    positives = tf.linalg.diag_part(similarity)
    nominator = tf.exp(positives/temperature)
    mask = tf.ones_like(similarity)-tf.eye(batch_size,batch_size)
    denominator = tf.reduce_sum(mask*tf.exp(similarity/temperature),axis=0)
    loss = -tf.reduce_mean(tf.math.log(nominator/denominator))

    return loss
#################################
# Create the twin encoders with shared weights#
#################################
input_1 = layers.Input(input_shape)
input_2 = layers.Input(input_shape)

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

output1 = [tower_1 ,tower_2 ]

model_HS = keras.Model(inputs=[input_1, input_2], outputs=output1)
model_HS.summary()    
##########################################
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
##########################################
#### Define the training loop for the network ###########
##########################################
def training_loop(model,cost,x_train,epochs=10,learning_rate=0.001,batch_size=512):
    train_loss,val_loss=[],[]
    #Weights=[]
    train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train)).shuffle(x_train.shape[0]).batch(batch_size) 
    epoch_loss_avg = tf.keras.metrics.Mean()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = []
    for epoch in range(epochs):
    # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_ds):
            
            with tf.GradientTape() as tape:
                logits1 = model([x_batch_train,augment(x_batch_train)], training=True)
                
                loss_value = cost(logits1[0],logits1[1])
            epoch_loss_avg.update_state(loss_value)     
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            sys.stdout.write('\r'+'step %s/%s:  loss = %2.5f '%(step+1,len(train_ds),float(epoch_loss_avg.result())))

        tf.print('|   Epoch {:2d}:  Loss: {:2.5f}  '.format(epoch+1,epoch_loss_avg.result()))  
        loss.append(epoch_loss_avg.result())
    return loss    
####################################################
###### Fit the network to the training data #####################
loss_ = training_loop(model_HS,contrastive_loss,x_train,epochs=epoch,learning_rate=lr_schedule,batch_size=batch_size)      
########################################################
np.savetxt('loss_CLR.txt',loss_)


###########Train Linear classifier ###################
l1 = Dense(128, activation="linear")(embedding_network(input_1))
l1 =  Dense(1, activation="sigmoid")(l1)
dnn = keras.Model(inputs=input_1, outputs=l1)
dnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=tf.keras.losses.binary_crossentropy,metrics=['accuracy'])
#########################################
history_dnn=dnn.fit(X_train,y_train,batch_size=batch_size, epochs=epoch, shuffle=True,verbose=1)
np.savetxt('LCT_acc.txt',history_dnn.history['accuracy'])
#############################################
#########  Test the linear classifier and evaluate the network#
#############################################
scores_dnn = dnn.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (dnn.metrics_names[1], scores_dnn[1]*100))
score=dnn.predict(X_test);
fpr_dnn, tpr_dnn,_=roc_curve(y_test.ravel(),score.ravel());
from sklearn.metrics import auc
auc = auc(fpr_dnn, tpr_dnn)
print(f'AUC: {auc}')
#######################################


