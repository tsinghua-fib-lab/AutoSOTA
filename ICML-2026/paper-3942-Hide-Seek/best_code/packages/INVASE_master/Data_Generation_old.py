"""
Generating Synthetic Data for Synthetic Examples
There are 6 Synthetic Datasets

X ~ N(0,I) where d = 100

Y = 1/(1+logit)
- Syn1: logit = exp(X1 * X2)
- Syn2: logit = exp(X3^2 + X4^2 + X5^2 + X6^2 -4)
- Syn3: logit = -10 sin(2 * X7) + 2|X8| + X9 + exp(-X10) - 2.4
- Syn4: If X11 < 0, Syn1, X11 >= Syn2
- Syn4: If X11 < 0, Syn1, X11 >= Syn3
- Syn4: If X11 < 0, Syn2, X11 >= Syn3
""" 
#%% Necessary packages
import numpy as np 

#%% X Generation
def generate_X (n=10000):
    
    X = np.random.randn(n, 11)
    
    return X

#%% Basic Label Generation (Syn1, Syn2, Syn3)
'''
X: Features
data_type: Syn1, Syn2, Syn3
'''
def Basic_Label_Generation(X, data_type):
    
    # number of samples
    n = len(X[:,0])
    
    # Logit computation
    # 1. Syn1
    if (data_type == 'Syn1'):
        logit = np.exp(X[:,0]*X[:,1])
        
    # 2. Syn2
    elif (data_type == 'Syn2'):       
        logit = np.exp(np.sum(X[:,2:6]**2, axis = 1) - 4.0) 
        
    # 3. Syn3
    elif (data_type == 'Syn3'):
        logit = np.exp(-10 * np.sin(0.2*X[:,6]) + abs(X[:,7]) + X[:,8] + np.exp(-X[:,9])  - 2.4) 
        
    # P(Y=1|X) & P(Y=0|X)
    prob_1 = np.reshape( (1 / (1+logit)), [n,1])
    prob_0 = np.reshape( (logit / (1+logit)), [n,1])
    
    # Probability output
    prob_y = np.concatenate((prob_0,prob_1), axis = 1)
    
    # Sampling from the probability
    y = np.zeros([n,2])
    y[:,0] = np.reshape(np.random.binomial(1, prob_0), [n,])
    y[:,1] = 1-y[:,0]

    return y, prob_y
    
#%% Complex Label Generation (Syn4, Syn5, Syn6)

def Complex_Label_Generation(X, data_type):
    
    # number of samples
    n = len(X[:,0])
    
    # Logit generation
    # 1. Syn4
    if (data_type == 'Syn4'):
        logit1 = np.exp(X[:,0]*X[:,1])
        logit2 = np.exp(np.sum(X[:,2:6]**2, axis = 1) - 4.0) 
    
    # 2. Syn5
    elif (data_type == 'Syn5'):
        logit1 = np.exp(X[:,0]*X[:,1])
        logit2 = np.exp(-10 * np.sin(0.2*X[:,6]) + abs(X[:,7]) + X[:,8] + np.exp(-X[:,9])  - 2.4) 
    
    # 3. Syn6
    elif (data_type == 'Syn6'):
        logit1 = np.exp(np.sum(X[:,2:6]**2, axis = 1) - 4.0) 
        logit2 = np.exp(-10 * np.sin(0.2*X[:,6]) + abs(X[:,7]) + X[:,8] + np.exp(-X[:,9])  - 2.4) 

    # Based on X[:,10], combine two logits        
    idx1 = (X[:,10]< 0)*1
    idx2 = (X[:,10]>=0)*1
    
    logit = logit1 * idx1 + logit2 * idx2
        
    # P(Y=1|X) & P(Y=0|X)
    prob_1 = np.reshape( (1 / (1+logit)), [n,1])
    prob_0 = np.reshape( (logit / (1+logit)), [n,1])
    
    # Probability output
    prob_y = np.concatenate((prob_0,prob_1), axis = 1)
    
    # Sampling from the probability
    y = np.zeros([n,2])
    y[:,0] = np.reshape(np.random.binomial(1, prob_0), [n,])
    y[:,1] = 1-y[:,0]

    return y, prob_y

#%% Ground truth Variable Importance

def Ground_Truth_Generation(X, data_type):

    # Number of samples and features
    n = len(X[:,0])
    d = len(X[0,:])

    # Output initialization
    out = np.zeros([n,d])
    
    # Index
    if (data_type in ['Syn4','Syn5','Syn6']):        
        idx1 = np.where(X[:,10]< 0)[0]
        idx2 = np.where(X[:,10]>=0)[0]
        out[:,10] = 1
    
    # For each data_type
    # Simple
    if (data_type == 'Syn1'):
        out[:,:2] = 1
    elif (data_type == 'Syn2'):
        out[:,2:6] = 1
    elif (data_type == 'Syn3'):
        out[:,6:10] = 1
        
    # Complex
    elif (data_type == 'Syn4'):        
        out[idx1,:2] = 1
        out[idx2,2:6] = 1
    elif (data_type == 'Syn5'):        
        out[idx1,:2] = 1
        out[idx2,6:10] = 1
    elif (data_type == 'Syn6'):        
        out[idx1,2:6] = 1
        out[idx2,6:10] = 1
        
    return out

    
#%% Generate X and Y
'''
n: Number of samples
data_type: Syn1 to Syn6
out: Y or Prob_Y
'''    
    
def generate_data(n=10000, data_type='Syn1', seed = 0, out = 'Y'):

    # For same seed
    np.random.seed(seed)

    # X generation
    X = generate_X(n)

    # Y generation
    if (data_type in ['Syn1','Syn2','Syn3']):
        Y, Prob_Y = Basic_Label_Generation(X, data_type)
        
    elif (data_type in ['Syn4','Syn5','Syn6']):
        Y, Prob_Y = Complex_Label_Generation(X, data_type)
    
    # Output
    if out == 'Prob':
        Y_Out = Prob_Y
    elif out == 'Y':
        Y_Out = Y
        
    # Ground truth
    Ground_Truth = Ground_Truth_Generation(X, data_type)
        
    return X, Y_Out, Ground_Truth
    