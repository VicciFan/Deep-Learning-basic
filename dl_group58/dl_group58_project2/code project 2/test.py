from optimizer import *
import numpy as np
import math
import time

'''
First generate the data,and compute the labels."train_value" is the value for 1 means in the circle, 0 means not in the circle.
"train_labels" is the one hot label of train_value for testing classification with softmax and cross_entropy
'''
x = np.random.uniform(0,1,1000)
x.shape=(x.shape[0],1)
y = np.random.uniform(0,1,1000)
y.shape=(y.shape[0],1)
train_data=np.column_stack((x, y))
r=((x-0.5)**2+(y-0.5)**2)**0.5
train_labels=r<1/(2*math.pi)**0.5
# 0 and 1 label
train_value=train_labels.copy().astype(np.int8)
# one hot label
train_labels.shape=train_labels.shape[0],
train_labels_index=train_labels.astype(np.int)
train_labels=np.zeros((1000,2))
train_labels[np.arange(1000), train_labels_index] = 1 
      
x_test = np.random.uniform(0,1,1000)
x_test.shape=(x_test.shape[0],1)
y_test = np.random.uniform(0,1,1000)
y_test.shape=(y_test.shape[0],1)
test_data=np.column_stack((x_test, y_test))
r=((x_test-0.5)**2+(y_test-0.5)**2)**0.5
test_labels=r<1/(2*math.pi)**0.5
# 0 and 1 label
test_value=test_labels.copy().astype(np.int8)
# one hot label
test_labels.shape=test_labels.shape[0],
test_labels_index=test_labels.astype(np.int)
test_labels=np.zeros((1000,2))
test_labels[np.arange(1000), test_labels_index] = 1 

# Now we define two functions to compute the error rate
# Compute the error rate for one hot label
def compute_error_onehot(model,step,data,label):
    pr=Predict(model)
    predict_labels=pr.batch_predict(data)
    predict_labels=np.where(predict_labels>0.5,1,0)

    err_rate=np.sum(abs(label-predict_labels))/2.0/1000.0
    return err_rate,predict_labels

# Compute the error rate for label of 0,1 
def compute_error_normal(model,step,data,value):
    pr=Predict(model)
    predict_labels=pr.batch_predict(data)
    predict_labels=np.where(predict_labels>0.5,1,0)
    
    err_rate=np.sum(abs(value-predict_labels))/1000.0
    return err_rate,predict_labels

def compute_error_rate(model,step,train_data,train_labels,test_data,test_labels,one_hot):
    if one_hot is True:
        err_train, _ = compute_error_onehot(model,step,train_data,train_labels)
        err_test, _ = compute_error_onehot(model,step,test_data,test_labels)
    else:
        err_train, _ = compute_error_normal(model,step,train_data,train_labels)
        err_test, _ = compute_error_normal(model,step,test_data,test_labels)
    return err_train,err_test
        
#########################################################################################################
print ("Test with one hot labels with softmax and cross entropy")  
'''
Build the network and predict the data
 
You can also write like this:
    
l1=Inputholder()
l2=Linear(25,l1)
l3=ReLU(l2)
l4=Linear(25,l3)
l5=ReLU(l4)
l6=Linear(25,l5)
l7=ReLU(l6)
l8=Linear(2,l7)
l9=Softmax(l8)
err=LossCrossEntropy(l9)


Or you can write with this Sequential class
'''
l1 = Sequential(Inputholder(),\
                Linear(25),\
                ReLU(),\
                Linear(25),\
                ReLU(),\
                Linear(25),\
                ReLU(),\
                Linear(2),\
                Softmax(),\
                LossCrossEntropy())


op=Optimizer(train_data,train_labels,l1,5000,lr=0.1,method="SGD",print_curve=True,batch_size=500,lr_options="t_frac_decay",\
             decay_rate=0.01,test_function=compute_error_rate,test_function_args=(train_data,train_labels,test_data,test_labels,True),test_interval=10)

start = time.clock()
op.optimize()
elapsed = (time.clock() - start)
print("Time used:",elapsed)

'''
Now we print a plot for the train error rate and the test error rate
'''
train_err_list = np.array(list(op.test_return))[:,0]
test_err_list = np.array(list(op.test_return))[:,1]
# Test interval is 10,whole steps is 5000,the step 0 is not computed,but the step 5000 is computed
x_value = np.arange(10,5010,10)
plt.plot(x_value,train_err_list,label="train error rate")
plt.plot(x_value,test_err_list,label="test error rate")
plt.title("Error rate curve")
plt.xlabel("step")
plt.ylabel("Error rate")
plt.xlim(xmin=10)
plt.legend()
plt.show()


err_train , predict_labels = compute_error_onehot(l1,0,train_data,train_labels)
err_test , predict_labels_test = compute_error_onehot(l1,0,test_data,test_labels)
print ("Error rate for train data")
print (str(err_train*100)[0:5]+"%")
print ("Error rate for test data")
print (str(err_test*100)[0:5]+"%")
print ("Visualize the classification results")
'''
Visualize the classification results
'''
print ("Visualize training data")
plt.figure(figsize=(6,6))
plt.title("Visualize training data")
ax = plt.gca()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
circle1 = plt.Circle((0.5, 0.5), 1/math.sqrt(2*math.pi), color='lightpink',alpha=0.5)
ax.add_artist(circle1)
plt.scatter(x[predict_labels[:,1]==0],y[predict_labels[:,1]==0],color='lightgreen',marker=".")  
plt.scatter(x[predict_labels[:,1]==1],y[predict_labels[:,1]==1],color='slateblue',marker=".")
ax.set_aspect(1)  
plt.show()
print ("Visualize testing data")
plt.figure(figsize=(6,6))
plt.title("Visualize testing data")
ax = plt.gca()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
circle1 = plt.Circle((0.5, 0.5), 1/math.sqrt(2*math.pi), color='lightpink',alpha=0.5)
ax.add_artist(circle1)
plt.scatter(x_test[predict_labels_test[:,1]==0],y_test[predict_labels_test[:,1]==0],color='lightgreen',marker=".")  
plt.scatter(x_test[predict_labels_test[:,1]==1],y_test[predict_labels_test[:,1]==1],color='slateblue',marker=".")
ax.set_aspect(1)  
plt.show()

#########################################################################################################
'''
Now we do the same thing with sigmoid function and MSE error. We do not use one hot labels this time.
'''
print ("Test with MSE error and sigmoid output")

'''
Build the network and predict the data

Also you can write like this:
    
l1=Inputholder()
l2=Linear(25,l1)
l3=Sigmoid(l2)
l4=Linear(25,l3)
l5=Sigmoid(l4)
l6=Linear(25,l5)
l7=Sigmoid(l6)
l8=Linear(1,l7)
l9=Sigmoid(l8)
err=LossMSE(l9)


Or you can write with this Sequential class
'''
l1 = Sequential(Inputholder(),\
                Linear(25),\
                Sigmoid(),\
                Linear(25),\
                Sigmoid(),\
                Linear(25),\
                Sigmoid(),\
                Linear(1),\
                Sigmoid(),\
                LossMSE())

op=Optimizer(train_data,train_value,l1,5000,lr=0.8,method="SGD",print_curve=True,batch_size=500,lr_options="exp_decay",\
             decay_rate=0.0002,test_function=compute_error_rate,test_function_args=(train_data,train_value,test_data,test_value,False),test_interval=10)

start = time.clock()
op.optimize()
elapsed = (time.clock() - start)
print("Time used:",elapsed)


'''
Now we print a plot for the train error rate and the test error rate
'''
train_err_list = np.array(list(op.test_return))[:,0]
test_err_list = np.array(list(op.test_return))[:,1]
# Test interval is 10,whole steps is 5000,the step 0 is not computed,but the step 5000 is computed
x_value = np.arange(10,5010,10)
plt.plot(x_value,train_err_list,label="train error rate")
plt.plot(x_value,test_err_list,label="test error rate")
plt.title("Error rate curve")
plt.xlabel("step")
plt.ylabel("Error rate")
plt.xlim(xmin=10)
plt.legend()
plt.show()


err_train , predict_labels = compute_error_normal(l1,0,train_data,train_value)
err_test , predict_labels_test = compute_error_normal(l1,0,test_data,test_value)
print ("Error rate for train data")
print (str(err_train*100)[0:5]+"%")
print ("Error rate for test data")
print (str(err_test*100)[0:5]+"%")
print ("Visualize the classification results")
'''
Visualize the classification results
'''
print ("Visualize training data")
plt.figure(figsize=(6,6))
plt.title("Visualize training data")
ax = plt.gca()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
circle1 = plt.Circle((0.5, 0.5), 1/math.sqrt(2*math.pi), color='lightpink',alpha=0.5)
ax.add_artist(circle1)
plt.scatter(x[predict_labels==0],y[predict_labels==0],color='lightgreen',marker=".")  
plt.scatter(x[predict_labels==1],y[predict_labels==1],color='slateblue',marker=".")
ax.set_aspect(1)  
plt.show()
print ("Visualize testing data")
plt.figure(figsize=(6,6))
plt.title("Visualize testing data")
ax = plt.gca()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
circle1 = plt.Circle((0.5, 0.5), 1/math.sqrt(2*math.pi), color='lightpink',alpha=0.5)
ax.add_artist(circle1)
plt.scatter(x_test[predict_labels_test==0],y_test[predict_labels_test==0],color='lightgreen',marker=".")  
plt.scatter(x_test[predict_labels_test==1],y_test[predict_labels_test==1],color='slateblue',marker=".")
ax.set_aspect(1)  
plt.show()

#########################################################################################################

'''
Now we use cross entropy loss again.But this time we use tanh as the inner activation layer.(In order to test tanh layer)
'''

print ("Test with one hot labels with softmax and cross entropy (tanh activation)")  
'''
Build the network and predict the data
 
You can also write like this:
    
l1=Inputholder()
l2=Linear(25,l1)
l3=Tanh(l2)
l4=Linear(25,l3)
l5=Tanh(l4)
l6=Linear(25,l5)
l7=Tanh(l6)
l8=Linear(2,l7)
l9=Softmax(l8)
err=LossCrossEntropy(l9)


Or you can write with this Sequential class
'''
l1 = Sequential(Inputholder(),\
                Linear(25),\
                Tanh(),\
                Linear(25),\
                Tanh(),\
                Linear(25),\
                Tanh(),\
                Linear(2),\
                Softmax(),\
                LossCrossEntropy())


op=Optimizer(train_data,train_labels,l1,5000,lr=0.03,method="SGD",print_curve=True,batch_size=500,lr_options="t_frac_decay",\
             decay_rate=0.0010,test_function=compute_error_rate,test_function_args=(train_data,train_labels,test_data,test_labels,True),test_interval=10)

start = time.clock()
op.optimize()
elapsed = (time.clock() - start)
print("Time used:",elapsed)


'''
Now we print a plot for the train error rate and the test error rate
'''
train_err_list = np.array(list(op.test_return))[:,0]
test_err_list = np.array(list(op.test_return))[:,1]
# Test interval is 10,whole steps is 5000,the step 0 is not computed,but the step 5000 is computed
x_value = np.arange(10,5010,10)
plt.plot(x_value,train_err_list,label="train error rate")
plt.plot(x_value,test_err_list,label="test error rate")
plt.title("Error rate curve")
plt.xlabel("step")
plt.ylabel("Error rate")
plt.xlim(xmin=10)
plt.legend()
plt.show()


err_train , predict_labels = compute_error_onehot(l1,0,train_data,train_labels)
err_test , predict_labels_test = compute_error_onehot(l1,0,test_data,test_labels)
print ("Error rate for train data")
print (str(err_train*100)[0:5]+"%")
print ("Error rate for test data")
print (str(err_test*100)[0:5]+"%")
print ("Visualize the classification results")
'''
Visualize the classification results
'''
print ("Visualize training data")
plt.figure(figsize=(6,6))
plt.title("Visualize training data")
ax = plt.gca()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
circle1 = plt.Circle((0.5, 0.5), 1/math.sqrt(2*math.pi), color='lightpink',alpha=0.5)
ax.add_artist(circle1)
plt.scatter(x[predict_labels[:,1]==0],y[predict_labels[:,1]==0],color='lightgreen',marker=".")  
plt.scatter(x[predict_labels[:,1]==1],y[predict_labels[:,1]==1],color='slateblue',marker=".")
ax.set_aspect(1)  
plt.show()
print ("Visualize testing data")
plt.figure(figsize=(6,6))
plt.title("Visualize testing data")
ax = plt.gca()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
circle1 = plt.Circle((0.5, 0.5), 1/math.sqrt(2*math.pi), color='lightpink',alpha=0.5)
ax.add_artist(circle1)
plt.scatter(x_test[predict_labels_test[:,1]==0],y_test[predict_labels_test[:,1]==0],color='lightgreen',marker=".")  
plt.scatter(x_test[predict_labels_test[:,1]==1],y_test[predict_labels_test[:,1]==1],color='slateblue',marker=".")
ax.set_aspect(1)  
plt.show()
