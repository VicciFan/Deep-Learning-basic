import numpy as np
import matplotlib.pyplot as plt
import dlc_bci as bci
from torch import cuda    
import torch
from torch.autograd import Variable

# Get the input data (train and test)       
train_input, train_target = bci.load(root='./data_bci',one_khz=True)

print(str(type(train_input)),train_input.size())
print(str(type(train_target)),train_target.size())

test_input, test_target = bci.load(root='./data_bci', one_khz=True,train=False)
print(str(type(test_input)), test_input.size())
print(str(type(test_target)), test_target.size())


# Define some layers that will be used after
# The permute layer is used to change the order of the dimension of the matrix
class Permute(torch.nn.Module):
    def __init__(self,order):
        super(Permute, self).__init__()
        self.order=order
    def forward(self, x):
        x = x.permute(*self.order)
        return x
		
# Flatten is used to keep only the first dimension and squeeze all the other dimensions
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        N=list(x.size())[0]
        x = x.contiguous()
        x = x.view(N,-1)
        return x
# Get the numpy values for the Tensors    
def evalTensor(x):
    # If the data is on GPU, first put back to CPU
    if cuda.is_available():
        x = x.cpu()
    return x.numpy()

	

# This is the index we choose to use	
index = [16,20,24,17,9,18,23,10,4 ,5 ,25, 7, 3]
index = np.array(index)

# Get the useful training and testing channels
train_input = train_input[:,index,:]
test_input = test_input[:,index,:]

# N is the number of training samples, C is the number of the channels we use
# W is the width of each channel, we take 300 window now 
# N_test is the number of testing samples
N, C, W, N_test= 316, len(index), 300, 100


# Now we do a normalization. We subtract the data by its mean and divide it by it standard deviation
# Notice we do this for every channels respectively
mu_list = []
sigma_list = []
data_train = train_input.numpy()
for i in range(len(index)):
    mu=np.mean(data_train[:,i,:])
    sigma = np.std(data_train[:,i,:]) 
    mu_list.append(mu)
    sigma_list.append(sigma)
    data_train[:,i,:] = (data_train[:,i,:]-mu)/sigma

# Get the last 300 data points    
data_train = data_train[:,:,200:500]
train_input = torch.FloatTensor(data_train)

# Do the same thing for test data. The mean and the standard deviation is computed using the the train data

data_test = test_input.numpy()
for i in range(len(index)):
    data_test[:,i,:] = (data_test[:,i,:]-mu_list[i])/sigma_list[i]

# Get the last 300 data points 
data_test = data_test[:,:,200:500]
test_input = torch.FloatTensor(data_test)


# Define the model
model = torch.nn.Sequential(
          Permute((0,3,1,2)),
          torch.nn.Conv2d(1,8,(1,7)),
          torch.nn.BatchNorm2d(8),
          Permute((0,2,1,3)),
          torch.nn.Conv2d(C,1,(1,1)),
          torch.nn.BatchNorm2d(1),
          torch.nn.Dropout(0.3),                    
          torch.nn.ELU(),
          torch.nn.MaxPool2d((1,3)),
          torch.nn.Conv2d(1,1,(1,3)),
          torch.nn.BatchNorm2d(1),
          torch.nn.Dropout(0.4),
          torch.nn.ELU(),
          torch.nn.MaxPool2d((1,3)),
          torch.nn.Conv2d(1,1,(1,3)),
          torch.nn.BatchNorm2d(1),
          torch.nn.Dropout(0.5),
          torch.nn.ELU(),
          torch.nn.MaxPool2d((1,3)),
          Flatten(),
          torch.nn.BatchNorm1d(80),
          torch.nn.Linear(80, 2)
        )

# Use cross entropy as a loss
loss_fn = torch.nn.CrossEntropyLoss(size_average=True)

# If GPU is available, train it in GPU
if cuda.is_available():
    model.cuda()
    loss_fn.cuda()
    train_input,train_target = train_input.cuda(), train_target.cuda()
    test_input,test_target = test_input.cuda(), test_target.cuda()

# Add a 1 dimension to the last to have a channel input for convolution layers
train_input = train_input.contiguous()
test_input = test_input.contiguous()
x = Variable(train_input.view(N,C,W,1))
x_test = Variable(test_input.view(N_test,C,W,1))
y = Variable(train_target, requires_grad=False)    
    
# Use adam as a optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

loss_list=[]
err_rate_train = []
err_rate_test = []

# Define the epochs to train and the batch size
nb_epochs = 1500
batch_size = 79*2

batch_index = list(range(316))

# Define a scheduler to change the learning rate
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250,400,800], gamma=0.75)

# The main loop
for k in range (nb_epochs):
    for b in range (0 , train_input.size(0),batch_size):
	    
        # Choose the data to use in this step
        if cuda.is_available():
            choose = torch.cuda.LongTensor(np.random.choice(batch_index,batch_size))
        else:
            choose = torch.LongTensor(np.random.choice(batch_index,batch_size))
        batch_x = x[choose,:,:,:]
        batch_y = y[choose]

        # Change to training mode
        model.train()
		
        # Do forward pass
        y_pred = model(batch_x)
    
        # Compute the loss
        loss = loss_fn(y_pred, batch_y)
        print(k, loss.data[0])
        loss_list.append(loss.data[0])
		
        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()
      
        # Do backward pass
        loss.backward()
      
        # Update the weights
        optimizer.step()
		
        # Change the model to test mode
        model.eval()
		
        # Compute the error rate
        predict_train = model.forward(x)
        _ , predict_train = torch.max(predict_train.data, 1)
        err = np.mean((evalTensor(predict_train-train_target))!=0)
        err_rate_train.append(err)

        predict_test = model.forward(x_test)
        _ , predict_test = torch.max(predict_test.data, 1)
        err = np.mean((evalTensor(predict_test-test_target))!=0)
        err_rate_test.append(err)
        
        # Change the learning rate of the optimizer
        scheduler.step(k)
		
# Change the model to test mode	and compute the final learning rate
model.eval()  
predict_train = model.forward(x)
_ , predict_train = torch.max(predict_train.data, 1)
print ("Train_error rate:")
err = np.mean((evalTensor(predict_train-train_target))!=0)
print (err)

# Print the test error
predict_test = model.forward(x_test)
_ , predict_test = torch.max(predict_test.data, 1)
print ("Test_error rate:")
err = np.mean((evalTensor(predict_test-test_target))!=0)
print (err)

plt.figure(figsize=(12, 8))
plt.plot(range(len(loss_list)),loss_list,label="loss")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()
plt.figure(figsize=(12, 8))
plt.plot(range(len(loss_list)),err_rate_train,label="train_error")
plt.plot(range(len(loss_list)),err_rate_test,label="test_error")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Error Rate")
plt.show()


