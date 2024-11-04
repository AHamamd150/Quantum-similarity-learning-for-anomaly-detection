import pennylane as qml
import matplotlib.pyplot as plt
from pennylane.optimize import AdamOptimizer
from pennylane import numpy as np
import qutip
import sys
from sklearn.datasets import make_classification
from tqdm import tqdm
#####################################
x_train_1 = np.load('x0.npz',allow_pickle=True)['arr_0']
x_train_2 = np.load('x1.npz',allow_pickle=True)['arr_0']

x_test = np.load('x3.npz',allow_pickle=True)['arr_0']


print(f'Train shape:  {x_train_1.shape} , Test shape:  {x_test_1.shape}  ')
#######################
qubits = 6
num_layers= 4
batch_size=128
#####################################
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
dev = qml.device("default.qubit", wires=qubits+1)
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
        qml.CSWAP(wires=[0, int(k + 1), int(qubits/2 + k + 1)])
    qml.Hadamard(wires=0)

    return qml.expval(qml.PauliZ(0)) 
    
shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=qubits)
params = np.random.random(size=shape)
param_rand = np.copy(params)
## create the quanutm layer
def iterate_minibatches(inputs, targets, batch_size):

    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]
####################################
def overlaps(params,out1,out2):
    swap = 0
    for i in range(len(out1)): 
        output2 = np.concatenate([out1[i],out2[i]],0)
        output_2 = qcircuit(params,output2)
        swap += output_2 
    return swap/(len(out1))
    
def cost(weights, A, B):
    ab = overlaps(weights,A,B)
    return  ab + 0.1
#####################################
from pennylane.optimize import AdamOptimizer 
opt = AdamOptimizer(0.1)
#################################        
for it in range(epoch):
  q = 0
  for xbatch1,xbatch2 in iterate_minibatches(x_train_1,x_train_2, batch_size=batch_size):
    q+=1
    params,_,_= opt.step(cost,params,xbatch1,xbatch2)  
    sys.stdout.write('\r'+ 'Processing batch No: %s  '%(q))
  print(f'\n Run Epoch number: {it+1}')
####################################
def overlaps1(params,out1,out2):
   output2 = np.concatenate([out1,out2],0)
   output_2 = qcircuit(params,output2)
   swap = output_2 
   return swap  
########################################
x_train_1_new = x_train_1[:len(x_test)]
x_train_2_new = x_train_2[:len(x_test)]
probs=[]
for  i in range(len(x_test)):
   print(f'Processing event No: {i+1}')
   probs.append(overlaps1(params,x_test[i],x_train_1_new[i])- overlaps1(params,x_test[i],x_train_2_new[i]))
  
probs = 2*np.array(probs)-1
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test,probs)
AUC = metrics.auc(tpr, fpr)
print(AUC)

np.savetxt('tpr.txt',tpr)
np.savetxt('fpr.txt',fpr)
plt.plot(fpr,tpr)
plt.show()






