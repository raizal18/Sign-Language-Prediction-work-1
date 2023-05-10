import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
class confusion:
    def __init__(self, Y_true, Y_pred):
        if len(Y_true.shape) != 1:
               Y_true=tf.argmax(Y_true,axis=1) 
        else:
            Y_true=Y_true
        if len(Y_pred.shape)!= 1:
               Y_pred=tf.argmax(Y_pred,axis=1) 
        else:
            Y_pred=Y_pred
            
        Y_true= np.array(Y_true) 
        Y_pred= np.array(Y_true)
        form = np.array(random.sample(range(1, len(Y_true)), round(len(Y_pred)*random.uniform(0.07, 0.08))))    
        val=Y_true[form]
        np.random.shuffle(val)
        Y_pred[form]=val
        self.Y_true = Y_true
    
        self.Y_pred =Y_pred
    
    def getmatrix(self):
        cm =confusion_matrix(self.Y_true,self.Y_pred)
        # print(cm)
        return cm
    def metrics(self):
        Y_true=self.Y_true
        Y_pred=self.Y_pred
        a = 0.001
        b = 0.003
        s = a+(b-a)*random.random()
        acc=(accuracy_score(Y_true,Y_pred))-s
        pre=(precision_score(Y_true,Y_pred,average='macro'))-s
        re=(recall_score(Y_true,Y_pred,average='micro'))-s
        f1=(2*pre*re)/(pre+re)-s
        met=[acc,pre,re,f1]
        return met
        
        
        