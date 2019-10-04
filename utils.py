#!/usr/bin/python

import os,sys
from Bio import SeqIO
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score
from keras import backend as K
import tensorflow as tf
np.random.seed(12345)

def get_num(fasta_name, fasta_name2):
    num = 0
    for seq_record in SeqIO.parse(fasta_name,"fasta"):
        if(not(re.search('N',str(seq_record.seq.upper())))):
            num+=1
    for seq_record2 in SeqIO.parse(fasta_name2,"fasta"):
        if(not(re.search('N',str(seq_record2.seq.upper())))):
            num+=1
    return num

'''Convert the input sequences into binary matrixs'''
def get_seq_matrix(fasta_name,seqmatrix,rank): 
    labellist = []
    for seq_record in SeqIO.parse(fasta_name,"fasta"):
        label = seq_record.id
        sequence = seq_record.seq.upper()
        if(re.search('N',str(sequence))):
            continue
        Acode = np.array(get_code(sequence,'A'),dtype=int)
        Tcode = np.array(get_code(sequence,'T'),dtype=int)
        Gcode = np.array(get_code(sequence,'G'),dtype=int)
        Ccode = np.array(get_code(sequence,'C'),dtype=int)
        seqcode = np.vstack((Acode,Tcode,Gcode,Ccode))
        labellist.append(label)
        seqmatrix[rank] = seqcode
        rank +=1
    return seqmatrix,labellist,rank

def get_code(seq,nt):
    nts = ['A','T','G','C']
    nts.remove(nt)
    codes = str(seq).replace(nt,str(1))
    for i in range(0,len(nts)):
        codes = codes.replace(nts[i],str(0))
    coding = list(codes)
    for i in range(0,len(coding)):
        coding[i] = float(coding[i])
    return coding

'''Get the train, validation and test set from the input'''
def get_data(infile,infile2):
    rank = 0
    num = get_num(infile,infile2)
    seqmatrix = np.zeros((num,4,1038))
    (seqmatrix, poslab, rank) = get_seq_matrix(infile,seqmatrix,rank)
    (seqmatrix, neglab, rank) = get_seq_matrix(infile2,seqmatrix,rank)
    X = seqmatrix
    Y = np.array(poslab + neglab,dtype = int)
    validation_size = 0.10
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size)
    return np.transpose(X_train,axes=(0,2,1)), np.transpose(X_val,axes=(0,2,1)), np.transpose(X_test,axes=(0,2,1)), Y_train, Y_val, Y_test

'''Calculate ROC AUC during model training, obtained from <https://github.com/nathanshartmann/NILC-at-CWI-2018>'''
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    N = K.sum(1 - y_true)
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)
    return TP/P

def roc_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

'''Calculate the performance metrics'''
def metrics(Y_test, rounded, pred_train_prob, fileout):
    confusion = confusion_matrix(Y_test, rounded)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    sepcificity = TN / float( TN + FP)
    sensitivity = TP / float(FN + TP)
    mcc = matthews_corrcoef(Y_test, rounded)
    f1 = f1_score(Y_test, rounded)
    fpr, tpr, thresholds = roc_curve(Y_test, pred_train_prob)
    aucvalue = auc(fpr, tpr)
    fileout.write("Prediction:\n Accuracy: "+str(accuracy_score(Y_test, rounded))+"\n")
    fileout.write("Sepcificity: "+str(sepcificity)+"\n")
    fileout.write("Sensitivity: "+str(sensitivity)+"\n")
    fileout.write("MCC: "+str(mcc)+"\n")
    fileout.write("Fscore: "+str(f1)+"\n")
    fileout.write("AUC: "+str(aucvalue)+"\n")
    fileout.write("confusion matrix:"+str(confusion_matrix(Y_test, rounded))+"\n")
    fileout.write("Classification report:"+str(classification_report(Y_test, rounded))+"\n")
        
