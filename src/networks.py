from keras.layers import Input, concatenate, MaxPooling2D,Conv2D, Dropout, Cropping2D, Activation, Conv2DTranspose, Dense, Flatten,add
from keras.layers.normalization import BatchNormalization
import utilities as util
from keras.models import Model, Sequential
from keras.optimizers import Adam
from losses import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format("channels_last")

'''
get_crop_size:
	- in unet architecture encoding and decoding layers are connected
	-> encoding layers have to be cropped due to unpadded convolutions
  - cropping values simply calculated by shapes of layers
  - considering uneven croppings, by cropping one more unit from right/top 
arguments:
	- enc_layer: layer from encoding part of unet
	- dec_layer: layer from decoding part of unet
returns:
  - (ch1, ch2), (cw1, cw2): number of units to crop from height and width (bottom/top, left/right respectively)
'''

def get_crop_size(enc_layer, dec_layer):
  # width cropping: 
	cw = enc_layer._keras_shape[1] - dec_layer._keras_shape[1]
	if cw % 2 != 0:
		cw1, cw2 = int(cw/2), int(cw/2) + 1
	else:
		cw1, cw2 = int(cw/2), int(cw/2)

	ch = enc_layer._keras_shape[2] - dec_layer._keras_shape[2]

	if ch % 2 != 0:
		ch1, ch2 = int(ch/2), int(ch/2) + 1
	else:
		ch1, ch2 = int(ch/2), int(ch/2)

	return (ch1, ch2), (cw1, cw2)



'''
get_unet
    returns unet architecture with given modifications
arguments:
        - filtersize: size of filters of first layers
        - n_filters: number of filters of first layer
        - dropout: Bool, wether dropout layers should be added after ConvLayers
        - dropout_rate: if dropout should be used - percentage of dropped activations
        - batch_norm: Bool, wether batch_norm layers should be added after ConvLayers

        - default will give the original unet architecture
returns:
        - model: Keras model
        - model_label: String, to keep Network labeligng uniform
        - out_dim: size of output image
'''


def get_unet(img_dim = 256, filtersize = 3, n_filters = 32, dropout = False, dropout_rate = 0.5, batch_norm = False, cropping = False):

  reg = None

  inputs = Input((img_dim,img_dim,1))
  if cropping == False:
    padding = 'same'
  else:
    padding = 'valid'

  conv1 = Conv2D(n_filters, (filtersize, filtersize), padding=padding, kernel_regularizer =reg)(inputs)

  if batch_norm:
    conv1 = BatchNormalization()(conv1)
  conv1 = Activation('relu')(conv1)

  conv1 = Conv2D(n_filters, (3, 3), padding=padding, kernel_regularizer =reg)(conv1)
  if batch_norm:
    conv1 = BatchNormalization()(conv1)
  conv1 = Activation('relu')(conv1)
  #if dropout:
  #	conv1 = Dropout(dropout_rate)(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(n_filters*2, (3, 3), padding=padding, kernel_regularizer =reg)(pool1)
  if batch_norm:
    conv2 = BatchNormalization()(conv2)
  conv2 = Activation('relu')(conv2)
  conv2 = Conv2D(n_filters*2, (3, 3), padding=padding, kernel_regularizer =reg)(conv2)
  if batch_norm:
    conv2 = BatchNormalization()(conv2)
  conv2 = Activation('relu')(conv2)
  #if dropout:
  #	conv2 = Dropout(dropout_rate)(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(n_filters*4, (3, 3), padding=padding, kernel_regularizer =reg)(pool2)
  if batch_norm:
    conv3 = BatchNormalization()(conv3)
  conv3 = Activation('relu')(conv3)
  conv3 = Conv2D(n_filters*4, (3, 3), padding=padding, kernel_regularizer =reg)(conv3)
  if batch_norm:
    conv3 = BatchNormalization()(conv3)
  conv3 = Activation('relu')(conv3)
  if dropout:
    conv3 = Dropout(dropout_rate)(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = Conv2D(n_filters*8, (3, 3), padding=padding, kernel_regularizer =reg)(pool3)
  if batch_norm:
    conv4 = BatchNormalization()(conv4)
  conv4 = Activation('relu')(conv4)
  conv4 = Conv2D(n_filters*8, (3, 3), padding=padding, kernel_regularizer =reg)(conv4)
  if batch_norm:
    conv4 = BatchNormalization()(conv4)
  conv4 = Activation('relu')(conv4)

  if dropout:
    conv4 = Dropout(dropout_rate)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  '''
    using different padding here, because otherwise to many pixels would be cropped
  '''
  conv5 = Conv2D(n_filters*16, (3, 3), padding='same',kernel_regularizer =reg)(pool4)
  if batch_norm:
    conv5 = BatchNormalization()(conv5)
  conv5 = Activation('relu')(conv5)

  conv5 = Conv2D(n_filters*16, (3, 3), padding='same', kernel_regularizer =reg)(conv5)
  if batch_norm:
    conv5 = BatchNormalization()(conv5)
  conv5 = Activation('relu')(conv5)

  if dropout:
    conv5 = Dropout(dropout_rate)(conv5)

  up6 = Conv2DTranspose(n_filters*8, (2,2), strides = (2,2), padding='same')(conv5)
  if cropping:
    ch, cw = get_crop_size(conv4, up6)
    conv4 = Cropping2D(cropping = (ch, cw))(conv4)
  conc6 = concatenate([up6, conv4], axis = 3)
  conv6 = Conv2D(n_filters*8, (3, 3), padding=padding, kernel_regularizer =reg)(conc6)
  if batch_norm:
    conv6 = BatchNormalization()(conv6)
  conv6 = Activation('relu')(conv6)
  conv6 = Conv2D(n_filters*8, (3, 3), padding=padding, kernel_regularizer =reg)(conv6)
  if batch_norm:
    conv6 = BatchNormalization()(conv6)
  conv6 = Activation('relu')(conv6)
  if dropout:
    conv6 = Dropout(dropout_rate)(conv6)

  up7 = Conv2DTranspose(n_filters*4, (2,2), strides = (2,2), padding='same')(conv6)

  cw, ch = get_crop_size(conv3, up7)

  if cropping:
    conv3	= Cropping2D(cropping = (cw, ch ))(conv3)

  conc7 = concatenate([up7, conv3], axis=3)
  conv7 = Conv2D(n_filters*4, (3, 3), padding=padding, kernel_regularizer =reg)(conc7)
  if batch_norm:
    conv7 = BatchNormalization()(conv7)
  conv7 = Activation('relu')(conv7)

  conv7 = Conv2D(n_filters*4, (3, 3), padding=padding, kernel_regularizer =reg)(conv7)
  if batch_norm:
    conv7 = BatchNormalization()(conv7)
  conv7 = Activation('relu')(conv7)

  if dropout:
    conv7 = Dropout(dropout_rate)(conv7)

  up8 = Conv2DTranspose(n_filters*2, (2,2), strides = (2,2), padding='same')(conv7)

  if cropping:
    cw, ch = get_crop_size(conv2, up8)
    conv2	= Cropping2D(cropping = (cw,ch))(conv2)
  conc8 = concatenate([up8, conv2], axis=3)
  conv8 = Conv2D(n_filters*2, (3, 3), padding=padding, kernel_regularizer =reg)(conc8)
  if batch_norm:
    conv8 = BatchNormalization()(conv8)
  conv8 = Activation('relu')(conv8)

  conv8 = Conv2D(n_filters*2, (3, 3), padding=padding, kernel_regularizer =reg)(conv8)
  if batch_norm:
    conv8 = BatchNormalization()(conv8)
  conv8 = Activation('relu')(conv8)

  #if dropout:
  #	conv8 = Dropout(dropout_rate)(conv8)

  up9 = Conv2DTranspose(n_filters*8, (2,2), strides = (2,2), padding='same')(conv8)

  if cropping:
    cw, ch = get_crop_size(conv1, up9)
    conv1	= Cropping2D(cropping = (cw, ch))(conv1)

  conc9 = concatenate([up9, conv1], axis=3)
  conv9 = Conv2D(n_filters, (3, 3), padding=padding, kernel_regularizer =reg)(conc9)
  if batch_norm:
    conv9 = BatchNormalization()(conv9)
  conv9 = Activation('relu')(conv9)
  conv9 = Conv2D(n_filters, (3, 3), padding=padding,kernel_regularizer =reg)(conv9)
  if batch_norm:
    conv9 = BatchNormalization()(conv9)
  act9 = Activation('relu')(conv9)

  conv10 = Conv2D(1, (1, 1), activation='sigmoid')(act9)

  model = Model(inputs=inputs, outputs=conv10)

  model_name = util.get_model_name("unet_", filtersize, n_filters, dropout, batch_norm, cropping)
  if batch_norm:
    lr = 1e-2
  else:
    lr = 1e-5
  model.compile(optimizer=Adam(lr=lr), loss = dice_coef_loss, metrics=[dice_coef])

  return model, model_name, model.output_shape[1]

def residual_block(n_filters, filtersize, input):
  ''' 
  residual block of critic network in SCAN architecture: 
  arguments:
    n_filters: number of filters for Conv Layers
    filtersize: size of Conv kernels
    input: Keras Layer - input of ResBlock
  returns:
    act2: output of second activation Layer (== output of residual block)
  '''  
  conv1 = Conv2D(n_filters, (filtersize, filtersize), strides = (2,2), padding = 'same')(input)
  bn1 = BatchNormalization()(conv1,training=K.learning_phase())
  act1 = Activation('relu')(bn1)
  conv2 = Conv2D(n_filters, (filtersize,filtersize), padding = 'same')(act1)
  bn2 = BatchNormalization()(conv2)
  shortcut = Conv2D(n_filters, (1,1), strides = (2,2))(input)
  su1 = add([bn2, shortcut])
  bn3 = BatchNormalization()(su1, training=K.learning_phase())
  act2 = Activation('relu')(bn3)
  
  return act2    
  
def get_critic_scan(n_filters = 8, img_dim = 256,lr = 2e-4):
  
  '''
  get_critic_scan:
    returning critic network of scan architecture, similar to original (in terms of number and size of convolutional kernels), except not using pooling but strided convolutions instead
  arguments: 
    n_filters: number of filters in first layer of the network, afterwards filters are doubled in every new residual block
    img_dim: input dimension of image
    lr: learning rate of the model, default is the suggested value for Adam
  '''
  
  inp = Input((img_dim, img_dim, 1))
  
  rb1 = residual_block(n_filters, 7, inp)
  rb2 = residual_block(2*n_filters, 3, rb1)
  rb3 = residual_block(4*n_filters, 3, rb2)
  rb4 = residual_block(8*n_filters, 3, rb3)
  
  fl = Flatten()(rb4)
  fc = Dense(1, activation='sigmoid')(fl)  
  model = Model(inputs = inp, outputs = fc)
  model.compile(optimizer = Adam(lr), loss = 'binary_crossentropy')
  return model
    