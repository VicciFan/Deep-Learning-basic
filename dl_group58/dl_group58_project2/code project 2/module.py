from torch import FloatTensor
from torch import LongTensor
from initializer import *
class Module(object):
    def __init__(self):
        self.next_module=None
        self.gradient=None
        self.output=None
        self.phase="train"
    def forward(self , *input):
        # If the layer before did not do forward propagation,then force the layer before do the forward propagation first
        if (self.prev_module is not None) and (self.prev_module.output is None):
            self.prev_module.forward(False)
    def backward(self , *gradwrtoutput):
        # If the layer after did not do backward propagation,then force the layer after do the backward propagation first
        if (self.next_module is not None) and (self.next_module.gradient is None):
            self.next_module.backward(False) 
    def param(self):
        return  []
    def update(self,lr):
        pass
    def change_to_test(self):
        self.phase="test"
    def change_to_train(self):
        self.phase="train"
class Linear(Module):
    def __init__(self,output_dimension,prev_module=None,weight_initializer=gaussian_initializer(),bias_initializer=zeros_initializer()):
        super(Linear,self).__init__()
        self.__w=None
        self.__b=None
        self.prev_module=prev_module
        if self.prev_module is not None:
            self.prev_module.next_module=self
        self.output_dimension=output_dimension
        self.__b_gradient=0
        self.__w_gradient=0
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
    def forward(self,pro=False):
        '''
        Parameter:
            pro   :If True, automatically do the next layers forward propagation after finishing the computation of this layer.Default is False.
        '''
        super(Linear,self).forward()
        if self.__w is None:
            self.__w=FloatTensor(self.output_dimension,list(self.prev_module.output.size())[0])
            self.__w=self.weight_initializer.initialize(self.__w)
        if self.__b is None:
            self.__b=FloatTensor(self.output_dimension, 1).zero_()
            self.__b=self.bias_initializer.initialize(self.__b)
        # Compute the output by output=Wx+b
        self.output=self.__w.mm(self.prev_module.output)+self.__b
        if self.next_module is not None:
            if pro is True:
                self.next_module.forward(True)
    def param(self):
        return [(self.__w, self.__w_gradient),(self.__b,self.__b_gradient)]
    def backward(self,pro=False):
        '''
        Parameter:
            pro   :If True, automatically do the last layers backward propagation after finishing the computation of this layer.Default is False.
        '''
        # Make sure all the networks finished the forward process and the layers after have already finished the backward process.
        super(Linear,self).forward()
        super(Linear,self).backward()
        self.gradient=self.__w.t()
        if self.next_module is not None:
            # Gradient of b is just the gradient back propagate by the next layer
            self.__b_gradient=self.next_module.gradient
            # Gradient of w is the gradient back propagate by the next layer and the gradient comes from matrix multiply,which is x
            self.__w_gradient=self.next_module.gradient.mm(self.prev_module.output.t())
            # Gradient of this layer is the gradient back propagate by the next layer and the gradient comes from matrix multiply,which is W
            self.gradient=self.gradient.mm(self.next_module.gradient)
        if self.prev_module is not None:
            if pro is True:
                self.prev_module.backward(True)
    def update(self,lr):
        # Compute the average gradient in the gradient list and apply them to the parameters
        self.__w=self.__w-lr*self.__w_gradient/list(self.prev_module.output.size())[1]
        self.__b=self.__b-lr*self.__b_gradient.mean(1,True)
class Inputholder(Module):
    '''
    This layer is a holder of data.Every neural network should begin with this layer and use the input_value property to pass train or test data.
    '''
    def __init__(self,input_value=None):
        super(Inputholder,self).__init__()
        self.prev_module=None
        self.input_value=input_value
    def forward(self,pro=False):
        super(Inputholder,self).forward()
        self.output=FloatTensor(self.input_value).t()
        if self.next_module is not None:
            if pro is True:
                self.next_module.forward(True)
    def backward(self,pro=False):
        super(Inputholder,self).forward()
        super(Inputholder,self).backward()
        if self.next_module is not None:
            self.gradient=self.next_module.gradient
        '''if self.prev_module is not None:
            if pro is True:
                self.prev_module.backward(True)
        '''
class ReLU(Module):
    def __init__(self,prev_module=None):
        super(ReLU,self).__init__()
        self.prev_module=prev_module
        if self.prev_module is not None:
            self.prev_module.next_module=self
    def forward(self,pro=False):
        super(ReLU,self).forward()
        self.output=0.5*(self.prev_module.output+self.prev_module.output.abs())
        if self.next_module is not None:
            if pro is True:
                self.next_module.forward(True)
    def backward(self,pro=False):
        super(ReLU,self).forward()
        super(ReLU,self).backward()
        if self.next_module is not None:        
            self.gradient=0.5*(self.prev_module.output.sign()+1)
            self.gradient=self.gradient*self.next_module.gradient
        if self.prev_module is not None:
            if pro is True:
                self.prev_module.backward(True)
class Tanh(Module):
    def __init__(self,prev_module=None):
        super(Tanh,self).__init__()
        self.prev_module=prev_module
        if self.prev_module is not None:
            self.prev_module.next_module=self
    def forward(self,pro=False):
        super(Tanh,self).forward()
        self.output=self.prev_module.output.tanh()
        if self.next_module is not None:
            if pro is True:
                self.next_module.forward(True)
    def backward(self,pro=False):
        super(Tanh,self).forward()
        super(Tanh,self).backward()
        if self.next_module is not None:
            self.gradient=1-(self.prev_module.output.tanh())**2
            self.gradient=self.gradient*self.next_module.gradient
        if self.prev_module is not None:
            if pro is True:
                self.prev_module.backward(True)
class Sigmoid(Module):
    def __init__(self,prev_module=None):
        super(Sigmoid,self).__init__()
        self.prev_module=prev_module
        if self.prev_module is not None:
            self.prev_module.next_module=self
    def forward(self,pro=False):
        super(Sigmoid,self).forward()
        self.output=self.prev_module.output.sigmoid()
        if self.next_module is not None:
            if pro is True:
                self.next_module.forward(True)
    def backward(self,pro=False):
        super(Sigmoid,self).forward()
        super(Sigmoid,self).backward()
        if self.next_module is not None:
            self.gradient=self.output*(1-self.output)
            self.gradient=self.gradient*self.next_module.gradient
        if self.prev_module is not None:
            if pro is True:
                self.prev_module.backward(True)
class LossMSE(Module):
    def __init__(self,prev_module=None,target=None):
        super(LossMSE,self).__init__()
        self.prev_module=prev_module
        if self.prev_module is not None:
            self.prev_module.next_module=self
        if target is not None:
            self.target=FloatTensor(target)
    def forward(self,pro=False):
        super(LossMSE,self).forward()
        # Only do forward process in training time
        if self.phase =="train":
            self.output=((self.prev_module.output-self.target)**2).sum(0,True)
            if self.next_module is not None:
                if pro is True:
                    self.next_module.forward(True)
    def backward(self,pro=False):
        super(LossMSE,self).forward()
        super(LossMSE,self).backward()
        self.gradient=2*(self.prev_module.output-self.target)
        if self.next_module is not None:
            self.gradient=self.gradient*(self.next_module.gradient)
        if self.prev_module is not None:
            if pro is True:
                self.prev_module.backward(True)
class Softmax(Module):
    def __init__(self,prev_module=None):
        super(Softmax,self).__init__()
        self.prev_module=prev_module
        if self.prev_module is not None:
            self.prev_module.next_module=self
    def forward(self,pro=False):
        super(Softmax,self).forward()
        self.output=(self.prev_module.output-self.prev_module.output.max(0,True)[0]).exp()
        self.output=self.output/self.output.sum(0,True)
        if self.next_module is not None:
            if pro is True:
                self.next_module.forward(True)
    def backward(self,pro=False):
        super(Softmax,self).forward()
        super(Softmax,self).backward()     
        if self.next_module is not None:
            # Change axis to batch first
            self.gradient=self.output.t().clone()
            s=list(self.gradient.size())[0]
            # Rearange to contiguous to view
            self.gradient = self.gradient.contiguous()
            # Make a column vector and a row vector to compute the matrix for computing gradient
            self.gradient_1 = self.gradient.view(s,-1,1).clone()
            self.gradient = self.gradient.contiguous()
            self.gradient_2 = self.gradient.view(s,1,-1).clone()
            # Compute the gradient matrix
            self.gradient = self.gradient_1.bmm(self.gradient_2)
            self.gradient = -1*self.gradient.bmm(self.next_module.gradient.t().contiguous().view(s,-1,1))
            # Multiply also the gradient afterwards and add the extra gradient for i=j
            self.gradient = self.gradient.contiguous().view(s,-1).t()+self.next_module.gradient*self.output
        if self.prev_module is not None:
            if pro is True:
                self.prev_module.backward(True)
class LossCrossEntropy(Module):
    def __init__(self,prev_module=None,target=None):
        super(LossCrossEntropy,self).__init__()
        self.prev_module=prev_module
        if self.prev_module is not None:
            self.prev_module.next_module=self
        self.eps=1e-20
        if target is not None:
            self.target=FloatTensor(target)
    def forward(self,pro=False):
        super(LossCrossEntropy,self).forward()
        # Only do forward process in training time
        if self.phase =="train":
            self.output=-1*((self.prev_module.output+self.eps).log()*self.target).sum(0,True)
            if self.next_module is not None:
                if pro is True:
                    self.next_module.forward(True)
    def backward(self,pro=False):
        super(LossCrossEntropy,self).forward()
        super(LossCrossEntropy,self).backward()
        self.gradient=-1*(1/(self.prev_module.output+self.eps)*self.target)
        if self.next_module is not None:
            self.gradient=self.gradient*(self.next_module.gradient)
        if self.prev_module is not None:
            if pro is True:
                self.prev_module.backward(True)


class Sequential(Module):
    def __init__(self,*args):
        super(Sequential,self).__init__()
        self.prev_module = None
        # Get all the layers
        self.args = list(args)
        # Connect all the layers
        for lp,ln in zip(self.args[0:-1],self.args[1:]):
            ln.prev_module = lp
            lp.next_module = ln
        self.args[0].prev_module=self.prev_module
        self.args[-1].next_module=self.next_module
    def forward(self,pro=False):
        super(Sequential,self).forward()
        # Put the input_value to the inputholder
        if self.input_value is not None:
            self.args[0].input_value = self.input_value 
        # Put the target to the loss layer
        if self.target is not None:
            self.args[-1].target = self.target 
        # Do forward pass
        self.args[0].forward(True)
        # Get the output from the last layer
        self.output = self.args[-1].output
    def backward(self,pro=False):
        super(Sequential,self).forward()
        super(Sequential,self).backward()
        # Do backward
        self.args[-1].backward(True)
        # Get the gradient from the last layer
        self.gradient = self.args[0].gradient
    def param(self):
        params = {}
        # Find param from all the layers
        for index,l in enumerate(self.args):
            p = l.param()
            if p != []:
                # Put the type and the number of the layer as a index, and store the parameters
                params[type(l).__name__+" layer "+str(index)]= p
        return params
    def update(self,lr):
        for l in self.args:
            l.update(lr)
    def get_layer(self,index):
        # Return the middle layer
        return self.args[index]
    def change_to_test(self):
        self.phase="test"
        for l in self.args:
            l.change_to_test()
    def change_to_train(self):
        self.phase="train"
        for l in self.args:
            l.change_to_train()



















































































































































































































































































































