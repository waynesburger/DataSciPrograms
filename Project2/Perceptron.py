# -*- coding: utf-8 -*-
"""
Spyder Editor

This is not a temporary script file.
"""
#import matplotlib.pyplot as plotter
import numpy as np
import tensorflow as tf
import pandas as pand

print("it worked")

"""
------------- Obtain training data --------------------------------------------
"""

train_data = pand.read_csv('G:\\CS457\\Wayne_Project_2\\trainSineFunction.csv')
train_data.head()
train_data.shape

input_arr = train_data[['x','y']].as_matrix()
input_arr
target_arr = train_data[['target']].as_matrix()
target_arr

def getbatch(inptarr, targtarr, batchsize, ind):
    return inptarr[batchsize*ind:batchsize*(ind+1)], targtarr[batchsize*ind:batchsize*(ind+1)]

# here is simply a test of the getbatch function to see if it works
btX, btY = getbatch(input_arr, target_arr, 15, 0)
btX
btY

"""
----------------- Obtain testing data -----------------------------------------
"""
test_data = pand.read_csv('G:\\CS457\\Wayne_Project_2\\testSineFunction.csv')
test_data.head()
test_data.shape

test_input = test_data[['x','y']].as_matrix()
test_input
test_target = test_data[['target']].as_matrix()
test_target


"""
--------------------------------- Perceptron -----------------------------------------------
"""
learn_rt = 0.03

In_count = 2
L1_count = 4
L2_count = 3
Out_count = 1

inputPlace = tf.placeholder(tf.float32, [None, In_count])
outputPlace = tf.placeholder(tf.float32, [None, Out_count])

weights = {
        'L1_wt':  tf.Variable( tf.random_normal( [In_count, L1_count] ) ),
        'L2_wt':  tf.Variable( tf.random_normal( [L1_count, L2_count] ) ),
        'Out_wt': tf.Variable( tf.random_normal( [L2_count, Out_count] ) )
        }
biases = {
        'L1_bia':  tf.Variable( tf.random_normal( [L1_count] ) ),
        'L2_bia':  tf.Variable( tf.random_normal( [L2_count] ) ),
        'Out_bia': tf.Variable( tf.random_normal( [Out_count] ) )
        }

# this function is meant to connect all the neurons of the network together
def multilayer_perceptron(inputPlace, weights, biases):
    layer_1 = tf.add( tf.matmul( inputPlace, weights['L1_wt'] ), biases['L1_bia'] )
    layer_1 = tf.nn.sigmoid(layer_1, name="sigmoid")
    
    layer_2 = tf.add( tf.matmul( layer_1, weights['L2_wt'] ), biases['L2_bia'] )
    layer_2 = tf.nn.sigmoid(layer_2, name="sigmoid")
    
    out_layer = tf.matmul(layer_2, weights['Out_wt'])+biases['Out_bia']
    print(out_layer)
    
    return out_layer

multipercept = multilayer_perceptron(inputPlace, weights, biases)


# The loss operation is one that uses the mean squared method recommended in class
loss_op = tf.losses.mean_squared_error(labels=outputPlace, predictions=multipercept)
opti_op = tf.train.GradientDescentOptimizer(learning_rate=learn_rt).minimize(loss_op)
init_op = tf.global_variables_initializer()

"""
Training Session Live
"""
separate_str = "--------------\nThe Training Session is Live!!!!\n--------------------------------------"
print(separate_str)

epoch_count = 6

with tf.Session() as tensorSession:
    tensorSession.run(init_op)
    
    # Train the network
    for epoch in range(0, epoch_count):
        avg_loss = 0.0
        for batch in range(0,2):
            in_batch, out_batch = getbatch(input_arr, target_arr, 15, batch)
            guess = tensorSession.run(fetches=multipercept, feed_dict={inputPlace: in_batch, outputPlace: out_batch})
            print("The guess: {}".format(guess))
            thing, lossVal = tensorSession.run([opti_op,loss_op], feed_dict={inputPlace: in_batch, outputPlace: out_batch})
            #print(lossVal)
            avg_loss += lossVal/2
            #print(avg_loss)
        
        print("training round {}, average loss is {}".format(epoch+1, avg_loss) )
    
    #Test the network --------------------------

    #Print out the test results
    test_guess = tensorSession.run(fetches=multipercept, feed_dict={inputPlace: test_input, outputPlace: test_target})
    print("the official test predictions: {}".format(test_guess))

    #define the method to compute the accuracy
    predict = tf.nn.softmax(multipercept)
    correct_predict = tf.equal( tf.argmax(predict, 1), tf.argmax(outputPlace, 1) )
    #compute the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    accu_evaluated = accuracy.eval( {inputPlace: test_input, outputPlace: test_target} )
    print("Accuracy = {}".format(accu_evaluated) )
    print(outputPlace)
    

    tensorSession.close()
    
            

