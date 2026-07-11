'''
Instance-wise Variable Selection (INVASE)
for ICLR 2019 Conference
'''

#%% Necessary packages
# 1. Keras
from keras.layers import Input, Dense, Multiply
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

# 2. Others
import tensorflow as tf
import numpy as np

#%% Define PVS class
class PVS():
    
    # 1. Initialization
    '''
    x_train: training samples
    data_type: Syn1 to Syn 6
    '''
    def __init__(self, x_train, data_type, lamda):
        self.latent_dim1 = 100      # Dimension of actor (generator) network
        self.latent_dim2 = 200      # Dimension of critic (discriminator) network
        
        self.batch_size = 1000      # Batch size
        self.epochs = 10000         # Epoch size (large epoch is needed due to the policy gradient framework)
        self.lamda = lamda            # Hyper-parameter for the number of selected features

        self.input_shape = x_train.shape[1]     # Input dimension
        
        # Actionvation. (For Syn1 and 2, relu, others, selu)
        self.activation = 'relu' if data_type in ['Syn1','Syn2'] else 'selu'       

        # Use Adam optimizer with learning rate = 0.0001
        optimizer = Adam(0.0001)
        
        # Build and compile the discriminator (critic)
        self.discriminator = self.build_discriminator()
        # Use categorical cross entropy as the loss
        self.discriminator.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

        # Build the generator (actor)
        self.generator = self.build_generator()
        # Use custom loss (my loss)
        self.generator.compile(loss=self.my_loss, optimizer=optimizer)

    #%% Custom loss definition
    def my_loss(self, y_true, y_pred):
        
        # dimension of the features
        d = y_pred.shape[1]        
        
        # Put all three in y_true 
        # 1. selected probability
        sel_prob = y_true[:,:d]
        # 2. discriminator output
        dis_prob = y_true[:,d:(d+2)]
        # 3. ground truth
        y_final = y_true[:,(d+2):]        
        
        # A. Compute the rewards of the actor network
        Reward = tf.reduce_sum(y_final * tf.log(dis_prob + 1e-8), axis = 1)  

        # B. Policy gradient loss computation. 
        loss1 = Reward * tf.reduce_sum( sel_prob * K.log(y_pred + 1e-8) + (1-sel_prob) * K.log(1-y_pred + 1e-8), axis = 1) - self.lamda * tf.reduce_mean(y_pred, axis = 1)
        
        # C. Maximize the loss1
        loss = tf.reduce_mean(-loss1)

        return loss

    #%% Generator (Actor)
    def build_generator(self):

        model = Sequential()
        
        model.add(Dense(100, activation=self.activation, name = 's/dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape))
        model.add(Dense(100, activation=self.activation, name = 's/dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(Dense(self.input_shape, activation = 'sigmoid', name = 's/dense3', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()

        feature = Input(shape=(self.input_shape,), dtype='float32')
        select_prob = model(feature)

        return Model(feature, select_prob)

    #%% Discriminator (Critic)
    def build_discriminator(self):

        model = Sequential()
                
        model.add(Dense(200, activation=self.activation, name = 'dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape)) 
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(200, activation=self.activation, name = 'dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        model.add(Dense(2, activation ='softmax', name = 'dense3', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()
        
        # There are two inputs to be used in the discriminator
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')
        # 2. Selected Features
        select = Input(shape=(self.input_shape,), dtype='float32')         
        
        # Element-wise multiplication
        model_input = Multiply()([feature, select])
        prob = model(model_input)

        return Model([feature, select], prob)

    #%% Sampling the features based on the output of the generator
    def Sample_M(self, gen_prob):
        
        # Shape of the selection probability
        n = gen_prob.shape[0]
        d = gen_prob.shape[1]
                
        # Sampling
        samples = np.random.binomial(1, gen_prob, (n,d))
        
        return samples

    #%% Training procedure
    def train(self, x_train, y_train):

        # For each epoch (actually iterations)
        for epoch in range(self.epochs):

            #%% Train Discriminator
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            x_batch = x_train[idx,:]
            y_batch = y_train[idx,:]

            # Generate a batch of probabilities of feature selection
            gen_prob = self.generator.predict(x_batch)
            
            # Sampling the features based on the generated probability
            sel_prob = self.Sample_M(gen_prob)     
            
            # Compute the prediction of the critic based on the sampled features (used for generator training)
            dis_prob = self.discriminator.predict([x_batch, sel_prob])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch([x_batch, sel_prob], y_batch)

            #%% Train Generator
            # Use three things as the y_true: sel_prob, dis_prob, and ground truth (y_batch)
            y_batch_final = np.concatenate( (sel_prob, np.asarray(dis_prob), y_batch), axis = 1 )

            # Train the generator
            g_loss = self.generator.train_on_batch(x_batch, y_batch_final)

            #%% Plot the progress
            dialog = 'Epoch: ' + str(epoch) + ', d_loss (CE): ' + str(np.round(d_loss[0],4)) + ', d_loss (Acc): ' + str(d_loss[1]) + ', g_loss: ' + str(np.round(g_loss,4))
 
            if epoch % 100 == 0:              
                print(dialog)
    
    #%% Selected Features        
    def output(self, x_train):
        
        gen_prob = self.generator.predict(x_train)
        
        return np.asarray(gen_prob)


#%% Main Function
if __name__ == '__main__':
        
    # Data generation function import
    from Data_Generation import generate_data
    
    #%% Parameters
    # Synthetic data type    
    idx = 5
    data_sets = ['Syn1','Syn2','Syn3','Syn4','Syn5','Syn6']
    data_type = data_sets[idx]
    
    # No need to provide the number of relevant features at all!

    # Data output can be either binary (Y) or Probability (Prob)
    data_out_sets = ['Y','Prob']
    data_out = data_out_sets[0]
    
    # Number of Training and Testing samples
    train_N = 10000
    test_N = 10000
    
    # Seeds (different seeds for training and testing)
    train_seed = 0
    test_seed = 1
        
    #%% Data Generation (Train/Test)
    def create_data(data_type, data_out): 
        
        x_train, y_train, g_train = generate_data(n = train_N, data_type = data_type, seed = train_seed, out = data_out)  
        x_test,  y_test,  g_test  = generate_data(n = test_N,  data_type = data_type, seed = test_seed,  out = data_out)  
    
        return x_train, y_train, g_train, x_test, y_test, g_test
    
    x_train, y_train, g_train, x_test, y_test, g_test = create_data(data_type, data_out)

    #%% Hyperparameter
    lamda = 3

    # 1. PVS Class call
    PVS_Alg = PVS(x_train, data_type, lamda)
        
    # 2. Algorithm training
    PVS_Alg.train(x_train, y_train)    
    
    # 3. Get the selection probability on the testing set
    Sel_Prob_Test = PVS_Alg.output(x_test)
    
    # 4. Selected features
    score = 1.*(Sel_Prob_Test > 0.5)
    
    #%% Performance Metrics
    def performance_metric(score, g_truth):

        n = len(score)
        Temp_TPR = np.zeros([n,])
        Temp_FDR = np.zeros([n,])
        
        for i in range(n):
    
            # TPR    
            TPR_Nom = np.sum(score[i,:] * g_truth[i,:])
            TPR_Den = np.sum(g_truth[i,:])
            Temp_TPR[i] = 100 * float(TPR_Nom)/float(TPR_Den+1e-8)
        
            # FDR
            FDR_Nom = np.sum(score[i,:] * (1-g_truth[i,:]))
            FDR_Den = np.sum(score[i,:])
            Temp_FDR[i] = 100 * float(FDR_Nom)/float(FDR_Den+1e-8)
    
        return np.mean(Temp_TPR), np.mean(Temp_FDR), np.std(Temp_TPR), np.std(Temp_FDR)
    
    #%% Output
        
    TPR_mean, FDR_mean, TPR_std, FDR_std = performance_metric(score, g_test)
        
    print('TPR mean: ' + str(np.round(TPR_mean,1)) + '\%, ' + 'TPR std: ' + str(np.round(TPR_std,1)) + '\%, '  )
    print('FDR mean: ' + str(np.round(FDR_mean,1)) + '\%, ' + 'FDR std: ' + str(np.round(FDR_std,1)) + '\%, '  )
        
