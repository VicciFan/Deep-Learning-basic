from module import *
import random
import matplotlib.pyplot as plt
import math
import numpy as np
class Optimizer():
    '''
    This class implement different optimizer.To implement more,just define a function as def newOptimizer(self,lr).
    '''
    def __init__(self,train_data,train_labels,network,steps,lr=0.01, \
                 lr_options="constant",method="SGD",print_curve="False",batch_size=1,decay_rate=0.001, \
                                step_decay_length=100,print_interval=100,test_function=None,test_function_args=None,test_interval=100):
        '''
        Parameters
        train_data              :Numpy array of training data.One row is a sample.
        train_labels            :Numpy array of training labels.One row is a sample.
        network                 :The inputholder of the whole neural network.
        steps:                  :The number of steps to be trained.
        lr                      :Overall learning rate.Will be changed if the lr_options is not "constant".Default is 0.01.
        lr_options              :String to control the strategy of the learning rate.To see more details, please look the function "update_lr".
        method                  :Name of the optimizer.It matchs the functions in this class.Default is SGD.
        print_curve             :whether to print the curve of the loss.Default is false
        batch_size              :The batch size of SGD.Default is 1.
        decay_rate              :The parameter used when decaying the learning rate.Has different meanings for different "lr_options".
                                To see more details, please look the function "update_lr".
        step_decay_length       :The parameter used when "lr_options" is "step_decay". Default is 100.
                                To see more details, please look the function "update_lr".
        print_interval          :The interval of printing the loss.Default is 100.
        test_function           :This function will be excute by optimizer one time after every 'test_interval' steps. This function can be user 
                                defined to compute error rate for every test_interval steps or do visualization  for every test_interval steps.
                                It will at least have two arguments, the model and the step.The return of this function will be stored in the 
                                list 'test_return'. You can define this function by your self to have some test during the training procedure.
        test_function_args      :Some additional parameters that will be passed into the test_function.Must be a tuple.
        test_interval           :Interval to excute the test_function.Default is 100 steps.
        '''
        self.step=0
        self.train_data=train_data
        self.train_labels=train_labels
        self.network=network
        self.lr=lr
        self.lr_options=lr_options
        self.method=method
        self.steps=steps
        self.data_size=self.train_data.shape[0]
        self.print_curve=print_curve
        self.batch_size=batch_size
        self.decay_rate=decay_rate
        self.step_decay_length=step_decay_length
        while network.next_module is not None:
            network=network.next_module
        self.network_end=network
        self.print_interval=print_interval
        self.test_function=test_function
        self.test_function_args=test_function_args
        self.test_interval=test_interval
    def update_lr(self):
        '''
        This functions change the overall learning rate to learning rate of each steps.It implements weight decay.
        '''
        if self.lr_options=="constant":
            return self.lr
        if self.lr_options=="step_decay":
            s=self.step%self.step_decay_length
            return self.lr*self.decay_rate**s
        if self.lr_options=="exp_decay":
            return self.lr*math.exp(-self.decay_rate*self.steps)
        if self.lr_options=="t_frac_decay":
            return self.lr*1.0/(1+self.decay_rate*self.steps)
        return self.lr
    def optimize(self):
        '''
        This function is the mean entry for doing optimization.
        The for loop first compute the learning rate of this step,and use "eval" function to call the real entry of each optimizer.
        '''
        self.loss_curve=[]
        self.test_return=[]
        for i in range(self.step,self.steps):
            self.step=self.step+1
            lr=self.update_lr()
            cmd="self."+self.method+"(lr)"
            eval(cmd)
            if self.step%self.test_interval ==0:
                if self.test_function is not None:
                    if self.test_function_args is None:
                        self.test_return.append(self.test_function(self.network,self.step))
                    else:
                        self.test_return.append(self.test_function(self.network,self.step,*self.test_function_args))                                        
        if self.print_curve is True:
            plt.plot(range(len(self.loss_curve)),self.loss_curve)
            plt.title("curve for loss")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.show()
    def GD(self,lr):
        '''
        Just a whole batch version of SGD.All procedures are similar to SGD.
        '''
        
        network = self.network 
        # Change to train mode
        while network is not None:
            network.change_to_train()
            network=network.next_module
            
        # Choose the data,put it in the network
        data_choose=self.train_data
        self.network.input_value=FloatTensor(data_choose.astype(np.float))
        # Choose the label,put it in the network
        label_choose=self.train_labels
        self.network_end.target=FloatTensor(label_choose.astype(np.float)).t()
        # Do forward pass and backward pass
        self.network.forward(True)
        self.network_end.backward(True)
        network=self.network
        # Compute the error of this step
        err=self.network_end.output.mean()
        # Find all the layers in the network and use "update" to update the parameter
        while network.next_module is not None:
            network=network.next_module
            network.update(lr)
        # For last layer
        network.update(lr)
        if self.step%self.print_interval==0:
            print ("step: ",self.step)
            print ("error: ",err)
        self.loss_curve.append(err)
    def SGD(self,lr):
        '''
        A implemetation of SGD
        '''
        
        network = self.network 
        # Change to train mode
        while network is not None:
            network.change_to_train()
            network=network.next_module
            
        # Get the batch index for this step
        batch=random.sample(list(range(0, self.data_size)), self.batch_size)
        # Choose the data,put it in the network
        data_choose=self.train_data[batch,:]
        self.network.input_value=FloatTensor(data_choose.astype(np.float))
        # Choose the label,put it in the network
        label_choose=self.train_labels[batch,:]
        self.network_end.target=FloatTensor(label_choose.astype(np.float)).t()
        # Do forward pass and backward pass
        self.network.forward(True)
        self.network_end.backward(True)
        network=self.network
        # Compute the error of this step
        err=self.network_end.output.mean()
        # Find all the layers in the network and use "update" to update the parameter
        while network.next_module is not None:
            network=network.next_module
            network.update(lr)
        # For last layer
        network.update(lr)
        if self.step%self.print_interval==0:
            print ("step: ",self.step)
            print ("error: ",err)
        self.loss_curve.append(err)
class Predict():
    def __init__(self,network):
        self.network=network
        while network.next_module is not None:
            network=network.next_module
        self.network_end=network
    def batch_predict(self,test_batch):
        '''
        Do prediction of neural networks.
        '''
        # Put the data into the network
        self.network.input_value=FloatTensor(test_batch)
        network=self.network

        # Change to test mode
        while network is not None:
            network.change_to_test()
            network=network.next_module
            
        self.network.forward(True)
        
        # If it is a Sequential module , split the last loss layer to get the result
        # Otherwise , find the result in prev_module
        if type(self.network_end).__name__=="Sequential":
            re=self.network_end.get_layer(-2).output.t().numpy()
        else:
            re=self.network_end.prev_module.output.t().numpy()

        return re

        
        
            
            
        
        
        
        
        

