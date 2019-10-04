#!/usr/bin/python

from utils import *
import os,sys
import argparse
import h5py
import scipy.io
import numpy as np
from keras.optimizers import Adam
from keras.initializers import RandomUniform, RandomNormal, glorot_uniform, glorot_normal
from keras.models import Model
from keras.layers.core import  Dense, Dropout, Permute, Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras import regularizers
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, Input
from keras.layers.merge import multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
np.random.seed(12345)

'''Build the DeepLncCTCF model'''
def get_model(params):
    inputs = Input(shape = (1038, 4,))
    cnn_out = Convolution1D(int(params['filter']), int(params['window_size']),
    	kernel_initializer=params['kernel_initializer'], 
    	kernel_regularizer=regularizers.l2(params['l2_reg']), 
    	activation="relu")(inputs)
    pooling_out = MaxPooling1D(pool_size=int(params['pool_size']), 
    	strides=int(params['pool_size']))(cnn_out)
    dropout1 = Dropout(params['drop_out_cnn'])(pooling_out)
    cnn_out2 = Convolution1D(int(params['filter']), int(params['window_size']),
        kernel_initializer=params['kernel_initializer'],
        kernel_regularizer=regularizers.l2(params['l2_reg']),
        activation="relu")(dropout1)
    pooling_out2 = MaxPooling1D(pool_size=int(params['pool_size']),
        strides=int(params['pool_size']))(cnn_out2)
    dropout2 = Dropout(params['drop_out_cnn'])(pooling_out2)
    lstm_out = Bidirectional(LSTM(int(params['lstm_unit']), return_sequences=True, 
    	kernel_initializer=params['kernel_initializer'], 
    	kernel_regularizer=regularizers.l2(params['l2_reg'])), merge_mode = 'concat')(dropout2)
    a = Permute((2, 1))(lstm_out)
    a = Dense(lstm_out._keras_shape[1], activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_out = multiply([lstm_out, a_probs])
    attention_out = Lambda(lambda x: K.sum(x, axis=1))(attention_out)
    dropout2 = Dropout(params['drop_out_lstm'])(attention_out)
    dense_out = Dense(int(params['dense_unit']), activation='relu', 
    	kernel_initializer=params['kernel_initializer'], 
    	kernel_regularizer=regularizers.l2(params['l2_reg']))(dropout2)
    output = Dense(1, activation='sigmoid')(dense_out)
    model = Model(input=[inputs], output=output)
    adam = Adam(lr=params['learning_rate'],epsilon=10**-8)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[roc_auc])
    return model
 
'''Evaluate the model for 10 repetitions'''
def evaluation(infile,infile2,outfile):
    fileout = open(outfile, "w")
    for i in range(10):
        X_train,X_val,X_test,Y_train,Y_val,Y_test = get_data(infile,infile2)
        best = {'batch_size': 4.0, 'dense_unit': 112.0, 'drop_out_cnn': 0.4279770967884926, 'drop_out_lstm': 0.05028428952624636, 'filter': 208.0, 'kernel_initializer': 'random_uniform', 'l2_reg': 5.2164660610264974e-05, 'learning_rate': 0.00010199140620075788, 'lstm_unit': 64.0, 'pool_size': 4.0, 'window_size': 13.0}
        dnn_model = get_model(best)
        filepa = "bestmodel.r"+str(i)+".hdf5"
        checkpointer = ModelCheckpoint(filepath=filepa, verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        dnn_model.fit(X_train, Y_train, batch_size=2**int(best['batch_size']), epochs=100, shuffle=True, validation_data=(X_val,Y_val), callbacks=[checkpointer,earlystopper])
        predictions = dnn_model.predict(X_test)
        rounded = [round(x[0]) for x in predictions]
        pred_train_prob = predictions
        metrics(Y_test, rounded, pred_train_prob, fileout)

def main():
    parser = argparse.ArgumentParser(description="progrom usage")
    parser.add_argument("-f", "--fasta", type=str, help="positive instances")
    parser.add_argument("-n", "--negative", type=str, help="negatve instances")
    parser.add_argument("-o", "--out", type=str, help="prediction output")
    args = parser.parse_args()
    infile = args.fasta
    secondin = args.negative
    outfile = args.out
    evaluation(infile,secondin,outfile)

if __name__ == '__main__':
        main()
