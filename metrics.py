import numpy as np
import keras.backend as K
import tensorflow as tf

# функция минимальной стоимости обнаружения
def DCF(y_true, y_pred, cost_FN=10, cost_FP=1):
    TN = np.logical_and(K.eval(y_true) == 0, K.eval(K.round(y_pred)) == 0)
    TP = np.logical_and(K.eval(y_true) == 1, K.eval(K.round(y_pred) == 1))
    FP = np.logical_and(K.eval(y_true) == 0, K.eval(K.round(y_pred)) == 1)
    FN = np.logical_and(K.eval(y_true) == 1, K.eval(K.round(y_pred)) == 0)

    TN = K.sum(K.variable(TN))
    FP = K.sum(K.variable(FP))
    TP = K.sum(K.variable(TP))
    FN = K.sum(K.variable(FN))  

    FPR = FP/(TN+FP)
    FNR = FN/(FN+TP)
    DCF = (cost_FN * FNR + cost_FP * FPR)/(cost_FP+cost_FN)
    return(DCF)    

# метрика F1, содержащая precision и recall 
def F1(y_true, y_pred):

    TN = np.logical_and(K.eval(y_true) == 0, K.eval(K.round(y_pred)) == 0)
    TP = np.logical_and(K.eval(y_true) == 1, K.eval(K.round(y_pred) == 1))
    FP = np.logical_and(K.eval(y_true) == 0, K.eval(K.round(y_pred)) == 1)
    FN = np.logical_and(K.eval(y_true) == 1, K.eval(K.round(y_pred)) == 0)

    TN = K.sum(K.variable(TN))
    FP = K.sum(K.variable(FP))
    TP = K.sum(K.variable(TP))
    FN = K.sum(K.variable(FN))
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    F1 = 2* (Precision*Recall)/(Precision+Recall)  
    return(F1)

# Equal Error rate
def EER(y_true, y_pred):
    n_0 = tf.math.count_nonzero(tf.equal(y_true, 0), dtype=tf.float32) + tf.constant(K.epsilon())
    n_1 = tf.math.count_nonzero(tf.equal(y_true, 1), dtype=tf.float32) + tf.constant(K.epsilon())

    count_0 = tf.boolean_mask(y_pred, tf.equal(y_true, 0))
    count_1 = tf.boolean_mask(y_pred, tf.equal(y_true, 1))

    loop_vars = (tf.constant(0.0), tf.constant(1.0), tf.constant(0.0))
    start = lambda t, fpr, fnr: tf.greater_equal(fpr, fnr)
    body = lambda t, fpr, fnr: (
        t + 0.001,
        tf.divide(tf.math.count_nonzero(tf.greater_equal(count_0, t), dtype=tf.float32), n_0),
        tf.divide(tf.math.count_nonzero(tf.less(count_1, t), dtype=tf.float32), n_1)
    )

    t, fpr, fnr = tf.while_loop(start, body, loop_vars, back_prop=False)
    eer = (fpr + fnr) / 2
    return eer