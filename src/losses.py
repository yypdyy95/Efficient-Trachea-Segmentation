import keras.backend as K

'''
evaluation metrics: DICE-Coefficient
    - dice = 2 TP /(2 TP +FN + FP) = 2 TP / ((pred == True) + (GT == True))
    - use negative dice coefficient as loss function
'''

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return - dice_coef(y_true, y_pred)

    
'''
def TPVF(y_true, y_pred):

    y_true_f = K.round(K.flatten(y_true))
    y_pred_f = K.round(K.flatten(y_true))

    return K.sum(y_true_f * y_pred_f)/K.sum(y_true_f)

def FPVF(y_true, y_pred):

    y_true_f = K.round(K.flatten(y_true))
    y_pred_f = K.round(K.flatten(1 - y_true))

    return K.sum(y_true_f * y_pred_f)/K.sum(y_true_f)
'''


