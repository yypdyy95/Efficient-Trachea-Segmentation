import numpy as np
import matplotlib
matplotlib.use("ps")
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
import keras.backend as K
import time
from skimage.exposure import equalize_hist
import networks
from augmentation import augment_data
import utilities as util
import argparse
K.set_image_data_format('channels_last')

'''
workaround for buggy argparse with bools
from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
'''
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
'''
hyperparameters for training:
'''


'''
####################################################################
                    ARGUMENT PARSING
####################################################################
'''

batch_size = 64
validation_split = 0.2

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--filtersize', type = int, default = 3)
parser.add_argument('--n_filters', type=int, default=32)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--cropping', type=str2bool, default=False)
parser.add_argument('--hist_eq', type=str2bool, default=True)
parser.add_argument('--g_epochs', type=int, default=0, help="#augmentations" )
parser.add_argument('--f_iter', type=int, default = 1)
parser.add_argument('--filter', type=str2bool, default=True, help="using anisotropic diffusion")
parser.add_argument('--batch_norm', type=str2bool, default=True)
parser.add_argument('--f_kappa', type=float, default=0.01, help="diffusion constant")
parser.add_argument('--f_gamma', type=float, default=0.1, help="anisotropy exponent for diffusion filter")
parser.add_argument('--res_file', type=str, default="final_res.txt")
args = parser.parse_args()
print("Cropping = ", args.cropping)

dropout = (args.dropout_rate != 0.)

# adjusting batch size because of memory limitations
if args.n_filters == 16:
  batch_size = 32

if args.n_filters == 32:
  batch_size = 16

if args.n_filters == 64:
  batch_size = 8  

'''
####################################################################
                    LOADING MODEL AND DATA
####################################################################
'''

model, model_name, out_dim = networks.get_unet(filtersize = args.filtersize, n_filters = args.n_filters, dropout = dropout, batch_norm = args.batch_norm, cropping = args.cropping, dropout_rate = args.dropout_rate)

model.summary()

imgs, labels, test_imgs, test_labels = util.get_data(100)



'''
####################################################################
                    PREPROCESSING
####################################################################
'''

im_size = imgs.shape[2]
num_of_training_samples = imgs.shape[0]

all_imgs = np.concatenate([imgs, test_imgs])
all_labels = np.concatenate([labels, test_labels])
num_of_samples = all_imgs.shape[0]

i = 0
  
print("Started preprocessing")

tic = time.time()
  
if args.filter:
  for img in all_imgs:
    img = util.filter_anisotropic_diffusion(img, n_iter = 1, gamma = args.f_gamma, kappa=args.f_kappa)
    
    if args.hist_eq:
      img = equalize_hist(img)
  
all_imgs = all_imgs.reshape(num_of_samples,1,-1)

'''
normalize samples:
  - using normaly distributed values here different normalization techniques gave worse results 
'''

for i , img in enumerate(all_imgs):
  img /= (np.std(img)+1e-5)
  img -= np.mean(img)

toc = time.time()
print("finished preprocessing in {:.3f} seconds".format(toc - tic)) 
  
  
all_imgs = all_imgs.reshape(-1,im_size, im_size,1)
imgs 	= all_imgs[:num_of_training_samples]
labels 	= all_labels[:num_of_training_samples]
test_imgs = all_imgs[num_of_training_samples:]
test_labels = all_labels[num_of_training_samples:].reshape(-1,im_size, im_size,1)

del all_imgs
'''
extract validation data
'''

validation_samples = int(validation_split * imgs.shape[0])
validation_imgs = imgs[-validation_samples:].reshape(-1,im_size, im_size,1)
validation_labels = labels[-validation_samples:].reshape(-1,im_size,im_size,1)

imgs = imgs[:-validation_samples].reshape((-1,im_size,im_size,1))
labels = labels[:-validation_samples].reshape(-1,im_size, im_size,1)

'''
cropping output vectors
'''
  
inputs = imgs.reshape((-1, im_size, im_size,1))
val_im = validation_imgs.reshape((-1,im_size,im_size,1))
val_lab = validation_labels

del imgs
del validation_imgs
del validation_labels

'''
data augmentation using Keras image data generator
'''

print("\n  Training : \n ")

callbacks = [EarlyStopping(monitor='loss', min_delta = 0.001, patience=10, verbose=0)]

training_data, training_labels = augment_data(inputs, labels, epochs = args.g_epochs) 

  
hist = model.fit(training_data, training_labels, validation_data = (val_im, val_lab), batch_size = batch_size, epochs = args.epochs, callbacks = callbacks)

np.save('val_loss_'  + str(args.dropout_rate) + str(args.n_filters) + ".npy"  , hist.history['val_dice_coef'])

fig, ax = plt.subplots(1,1)

ax.plot(hist.history['dice_coef'], label = "$DICE_{train}$")
ax.plot(hist.history['val_dice_coef'], label="$DICE_{val}$")
ax.legend(loc = 'lower right')
ax.set_xlabel("Epoche")
ax.set_ylim([0.5,1.])
fig.savefig("hist" + str(args.dropout_rate) + str(args.n_filters) + ".png")

model.save("./../networks/" + model_name + ".h5")
 
test_score = model.evaluate(test_imgs, test_labels) 

print("Slicewise dice score on test samples: {:.3f}".format(test_score[1]))

'''
evaluation on test data:
'''

print("\n   Testing: \n")
tic = time.time()
predictions = model.predict(test_imgs)

toc = time.time()
prediction_time = toc-tic
print("predicting took " ,prediction_time , "seconds")
np.save("./../predictions/predictions"+model_name+ ".npy", predictions)
with open(args.res_file, 'a') as res_file:
  res_file.write("\n#################### \n#####################\n\n")
  
  for ar in args.__dict__:
      res_file.write(ar + "\t=\t" + str(args.__dict__[ar]) + "\n")
  res_file.write("===========================\n")
  #res_file.write("training DICE score: {}".format(hist['loss'][-1]))
DICE = util.test_evaluation(test_labels, predictions, args.res_file)

