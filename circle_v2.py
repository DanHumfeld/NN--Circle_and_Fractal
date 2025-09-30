#################################################################
# Code      Circle and Fractal
# Version   2.0
# Date      2022-10-11
# Author    Dan Humfeld, DanHumfeld@Yahoo.com
#
#################################################################
# Importing Libraries
#################################################################

# # Enable this section to hide all warnings and errors
import os
import logging
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import sys
stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
import absl.logging

import math
import random
import numpy as np
from numpy import savetxt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Multiply
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from time import time
start_time = time()

#################################################################
# Inputs 
#################################################################   
# Training Mode: Train new = 0, continue = 1, predictions only = 2
train_mode = 2
pretrain = True
# Transfer Mode: Starting = 0
transfer_mode = 0

# Path for model files
path_for_models = 'Working Models v2/'

# Epoch and batch size control
epochs = 400000
batch_max = 12000
epochs_pretrain = 1200

# Loss-related parameters
area_target = 2
initial_perimeter_target = 4.0 * math.sqrt(area_target)
perimeter_factor = 0.9998

# Optimizer for training: SGD, RSMProp, Adagrad, Adam, Adamax...
learning_rate = 0.0002 #0.0005
my_optimizer = optimizers.Adam(learning_rate)
initializer = 'glorot_uniform'

# Model hyper-parameters
nodes_per_layer = 32
model_layers = 3
model_count = 2

# Operating options
time_reporting = True

#################################################################
# File names
#################################################################
if (train_mode == train_mode):
    output_model_x = path_for_models + 'mode_' + str(transfer_mode) + '_x.h5'
    output_model_y = path_for_models + 'mode_' + str(transfer_mode) + '_y.h5'
    prediction_results_x = path_for_models + 'mode_' + str(transfer_mode) + '_x.csv'
    prediction_results_y = path_for_models + 'mode_' + str(transfer_mode) + '_y.csv'
    prediction_results_xy = path_for_models + 'mode_' + str(transfer_mode) + '_xy.csv'
    loss_history = path_for_models + 'mode_' + str(transfer_mode) + '_loss_history.csv'

if ((train_mode == 0) and (transfer_mode > 0)):
    input_model_x = path_for_models + 'mode_' + str(transfer_mode-1) + '_x.h5'
    input_model_y = path_for_models + 'mode_' + str(transfer_mode-1) + '_y.h5'
else:
    input_model_x = path_for_models + 'mode_' + str(transfer_mode) + '_x.h5'
    input_model_y = path_for_models + 'mode_' + str(transfer_mode) + '_y.h5'

if (train_mode == train_mode):
    input_model_file_names = []
    input_model_file_names.append(input_model_x)
    input_model_file_names.append(input_model_y)
    output_model_file_names = []
    output_model_file_names.append(output_model_x)
    output_model_file_names.append(output_model_y)
    prediction_results_file_names = []
    prediction_results_file_names.append(prediction_results_x)
    prediction_results_file_names.append(prediction_results_y)


#################################################################
# Building or Load Models - most of this code could be offloaded
#################################################################
def new_model(dimensions, layers, nodes, primary_activation, final_activation):
    model = Sequential()
    model.add(Dense(nodes, input_dim = dimensions, activation = primary_activation, kernel_initializer = initializer, bias_initializer = initializer))
    for layer_number in range(layers-1):
        model.add(Dense(nodes, activation = primary_activation, kernel_initializer = initializer, bias_initializer = initializer))
    model.add(Dense(1, activation = final_activation, kernel_initializer = initializer, bias_initializer = initializer))
    model.compile(loss = 'mse', optimizer = my_optimizer)
    return model

def load_models(models, file_names):
    for i in range(len(models)):
        models[i] = keras.models.load_model(file_names[i])
    return

def save_models(models, file_names):
    for i in range(len(models)):
        models[i].save(file_names[i])
    return

models = []
for model_number in range(model_count):                                        # Initialize new models, even if they will be overwritten via loading
    #models.append(new_model(1, model_layers, nodes_per_layer, 'elu', 'tanh'))
    models.append(new_model(1, model_layers, nodes_per_layer, 'elu', 'linear'))
    # For now just remember: x is 0, y is 1
if ((transfer_mode > 0) or (train_mode >= 1)):
    load_models(models, input_model_file_names)
save_models(models, output_model_file_names)

#################################################################
# Pretrain x and y to non-arbitrary, closed shape
#################################################################
if (train_mode in [0]) and (pretrain):
    print("Pretraining")
    min_loss = 100
    last_time = time()

    fixed_x = np.reshape([0.5, -0.5, -0.5,  0.5,  0.5],(5,1))
    fixed_y = np.reshape([0.5,  0.5, -0.5, -0.5,  0.5],(5,1))

    for i in range(0, epochs_pretrain):
        # Create tensors to feed to TF
        s_arr = np.sort(np.arange(5)/4)
        s_feed = np.column_stack((s_arr)) 
        s_feed = tf.Variable(s_feed.reshape(len(s_feed[0]),1), trainable=True, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape_1:
            # Watch parameters
            tape_1.watch(s_feed)

            # Define functions
            x = models[0]([s_feed])
            y = models[1]([s_feed])

            # Calculate x, y errors
            loss_x_list = k.square(x - fixed_x)
            loss_x = k.mean(loss_x_list)
            loss_y_list = k.square(y - fixed_y)
            loss_y = k.mean(loss_y_list)

            # Create losses
            losses = [loss_x, loss_y]
            loss_total = k.sum([loss_x, loss_y])

        # Train the models
        gradients = [tape_1.gradient(loss_total, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO) for model in models]
        for model_num in range(len(models)):
            my_optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients[model_num], models[model_num].trainable_variables) if grad is not None) 

        # Take a break and report
        i_loss = sum(losses)
        if i % 100 == 0:
            print("Step " + str(i) + " -------------------------------")
            print(["{:.3e}".format(k.get_value(loss)) for loss in losses])
            if (time_reporting):
                print("Calculation time for last period: ", "{:.0f}".format(round(time() - last_time, 0)))
            last_time = time()
            
            #Only save model if loss is improved
            if (min_loss > i_loss):
                min_loss = i_loss
                save_models(models, output_model_file_names)

    save_models(models, output_model_file_names)

#################################################################
# Main Code
#################################################################
if (train_mode in [0, 1]):
    print("Training mode = ", train_mode)
    print("Transfer mode = ", transfer_mode)

    min_loss = 100
    batch = batch_max
    last_time = time()
    perimeter_target = initial_perimeter_target

    for i in range(0, epochs):
        # Create tensors to feed to TF
        #s_arr = np.sort(np.random.uniform(0, 1, batch))
        s_arr = np.arange(batch)/(batch-1)
        s_feed = np.column_stack((s_arr)) 
        s_feed = tf.Variable(s_feed.reshape(len(s_feed[0]),1), trainable=True, dtype=tf.float32)

        zero_feed = np.column_stack([0])
        zero_feed = tf.Variable(zero_feed.reshape(len(zero_feed[0]),1), trainable=True, dtype=tf.float32)

        one_feed = np.column_stack([1])
        one_feed = tf.Variable(one_feed.reshape(len(one_feed[0]),1), trainable=True, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape_1:
            # Watch parameters
            tape_1.watch(s_feed)
            tape_1.watch(zero_feed)
            tape_1.watch(one_feed)

            # Define functions
            x = models[0]([s_feed])
            y = models[1]([s_feed])
            x0 = models[0]([zero_feed])
            x1 = models[0]([one_feed])
            y0 = models[1]([zero_feed])
            y1 = models[1]([one_feed])
            x_roll = tf.roll(x,-1,0)
            y_roll = tf.roll(y,-1,0)

            # Calculate area using shoelace formula, calcualte loss
            area_calculated = k.abs(k.sum(tf.multiply((x_roll + x), (y_roll - y)))/2.0)
            loss_area = k.square(area_target - area_calculated)
            
            # Calculate perimeter, loss
            perimeter = k.sum(k.sqrt(k.square(x - x_roll) + k.square(y - y_roll)))
            if ((i == 0) and (train_mode not in [0])):
                    perimeter_target = perimeter
            loss_perimeter = k.square(perimeter_target - perimeter)
            if (area_calculated / area_target > 0.8) and (area_calculated / area_target < 1.2):
                perimeter_target = perimeter_factor * k.get_value(perimeter)

            # Calculate closed perimeter loss term
            closed_loop_x = k.square(x1 - x0)
            closed_loop_y = k.square(y1 - y0)
            loss_closed_loop = k.sum([closed_loop_x, closed_loop_y])

            # Fixed end loss        # This term fixes the x(0)=y(0)=0 and then obviates the closed loop term by forcing x(1)=y(1)=0
            loss_fixed_end = k.sum([k.square(x0 - 0.0), k.square(y0 - 0.0), k.square(x1 - 0.0), k.square(y1 - 0.0)])
            #print(loss_fixed_end)

            # C2 loss term; the slope at s=0 must be the same as the slope at s=1 so there is no kink at s=0=1
            delta_x_0 = x[1] - x[0]
            delta_x_1 = x[-1] - x[-2]       # This isn't 0 and -1 because those are forced to be the same location
            delta_y_0 = y[1] - y[0]
            delta_y_1 = y[-1] - y[-2]
            loss_continuous_2 = tf.constant([0.0])
            if (delta_y_0 != 0) and (delta_y_1 != 0):
                slope_0 = delta_x_0 / delta_y_0
                slope_1 = delta_x_1 / delta_y_1
                loss_continuous_2 = k.square(slope_0 - slope_1)
            loss_continuous_2 = tf.reshape(loss_continuous_2,())
            #print(loss_continuous_2)

            # Calculate crossings loss term (necessary for fractals, probably)
            loss_crossings = 0.

            # Weight losses
            losses = [loss_perimeter, loss_area, loss_closed_loop, loss_fixed_end, loss_continuous_2] #, loss_crossings]
            loss_weightings = np.ones(len(losses))
            loss_weightings[0] = 1 #1.0e-3
            loss_weightings[2] = 1 #1.0e-1
            loss_weightings[3] = 1 #1.0e3
            loss_weightings[4] = 1 #0
            loss_total = sum([x*y for (x,y) in zip(losses, loss_weightings)])
            losses_weighted = [x*y for (x,y) in zip(losses, loss_weightings)]

        # Train the models
        #gradients_x = [tape_1.gradient(weighted_loss, models[0].trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO) for weighted_loss in losses_weighted]
        #gradients_y = [tape_1.gradient(weighted_loss, models[1].trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO) for weighted_loss in losses_weighted]
        gradients = [tape_1.gradient(loss_total, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO) for model in models]
        for model_num in range(len(models)):
            my_optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients[model_num], models[model_num].trainable_variables) if grad is not None) 

        # Take a break and report
        i_loss = sum(losses_weighted)
        #if i % 1 == 0:
        #if i % 10 == 0:
        if i % 100 == 0:
            print("Step " + str(i) + " -------------------------------")
            #print(losses)
            print(["{:.3e}".format(k.get_value(loss)) for loss in losses])
            #print(["{:.3e}".format(k.get_value(loss)) for loss in losses_weighted])
            print("Loss_tot: ", "{:.3e}".format(i_loss))
            print("Perimeter: ", "{:.3e}".format(k.get_value(perimeter)), "   Area: ", "{:.3e}".format(k.get_value(area_calculated)), "     Corner: (", "{:.3f}".format(k.get_value(x0[0][0])), ", ", "{:.3f}".format(k.get_value(y0[0][0])),")")
            if (time_reporting):
                print("Calculation time for last period: ", "{:.0f}".format(round(time() - last_time, 0)), "    Batch size: ", "{:.0f}".format(batch))
            last_time = time()
            
            #Only save model if loss is improved
            if (min_loss > i_loss):
                min_loss = i_loss
                save_models(models, output_model_file_names)
            
            # Adjust batch
            #batch = min(batch_max, max(batch_min, math.ceil(5 * i_loss**-1)))

    save_models(models, output_model_file_names)

#################################################################
# Predicting
#################################################################
# Inputs
nodes = batch_max
s_feed = np.arange(nodes)/(nodes-1)
report_time = []

x_output = np.reshape(models[0].predict([s_feed]),(nodes))
y_output = np.reshape(models[1].predict([s_feed]),(nodes))
results = np.column_stack((x_output, y_output))
np.savetxt(prediction_results_xy, results, delimiter=',') 


print("Job's done")
