import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
import time
import matplotlib 
matplotlib.use('ps')
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import networks
import utilities as util
from augmentation import augment_data
from keras.utils.generic_utils import Progbar
import argparse

'''
####################################################################
                    ARGUMENT PARSING
####################################################################
'''

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--pretrain_epochs', type=int, default = 5, help = "training epochs for segmentation network prior to real training")
parser.add_argument('--critic_epochs', type=int, default=1)
parser.add_argument('--filtersize', type = int, default = 3)
parser.add_argument('--n_filters', type=int, default=16)
parser.add_argument('--stacked_lr', type=float, default = 2e-4, help = "learning rate for stacked model")
parser.add_argument('--dropout_rate', type=float, default=0.)
parser.add_argument('--hist_eq', type=bool, default=True)
parser.add_argument('--g_epochs', type=int, default=0, help="#augmentations" )
parser.add_argument('--filter', type=bool, default=False, help="using anisotropic diffusion")
parser.add_argument('--c_filters', type = int, default = 16, help="#filters for critic network")
parser.add_argument('--batch_norm', type=bool, default=True)
parser.add_argument('--f_kappa', type=float, default=0.01, help="diffusion constant")
parser.add_argument('--f_gamma', type=float, default=0.1, help="anisotropy exponent for diffusion filter")
parser.add_argument('--res_file', type=str, default="scan_results.txt")
args = parser.parse_args()

dropout = (args.dropout_rate != 0.)

filtersize = args.filtersize
n_filters = args.n_filters

batch_size = 32
validation_split = 0.2

'''
####################################################################
                    LOADING MODEL AND DATA
####################################################################
'''

model, model_name, out_dim = networks.get_unet(filtersize = filtersize, n_filters = n_filters, dropout = dropout, batch_norm = args.batch_norm, dropout_rate = args.dropout_rate)

print(model.summary())

'''
loading reshaped samples:
'''
tic = time.time()

imgs, labels, test_imgs, test_labels = util.get_data(100)

toc=time.time()
print("\ ntime to load samples: {}".format(toc-tic))
	
print("\n loaded {} training samples and {} test samples".format(imgs.shape[0], test_imgs.shape[0]))

num_of_samples = test_imgs.shape[0]

img_size = imgs.shape[2]
num_of_training_samples = imgs.shape[0]

'''
concatenate samples to apply preprocessing to all in one loop

'''
all_imgs = np.concatenate([imgs, test_imgs])
all_labels = np.concatenate([labels, test_labels])
num_of_samples = all_imgs.shape[0]

'''
####################################################################
                    PREPROCESSING
####################################################################
'''
print("\n Started preprocessing")

tic = time.time()
'''
preprocessing: apply anisotropic diffusion and histogram equalisation on every image 

'''  
if args.filter:
  for img in all_imgs:
    img = util.filter_anisotropic_diffusion(img, n_iter = 1, gamma = args.f_gamma, kappa=args.f_kappa)
    
    if args.hist_eq:
      img = equalize_hist(img)

'''
to avoid issues with wrong calculation of mean/stddev reshape samples to 1x256*256 vector
'''      
all_imgs = all_imgs.reshape(num_of_samples,1,-1)

'''
normalize samples:
  - using normaly distributed values here (different normalization techniques, like maximum division, scaling between 1 and -1,... gave worse results) 
'''

for i , img in enumerate(all_imgs):
  img /= (np.std(img)+1e-5)
  img -= np.mean(img)

toc = time.time()
print("finished preprocessing in {:.3f} seconds".format(toc - tic)) 
  
  
all_imgs = all_imgs.reshape(-1,img_size, img_size,1)
imgs 	= all_imgs[:num_of_training_samples]
labels 	= all_labels[:num_of_training_samples]
test_imgs = all_imgs[num_of_training_samples:]
test_labels = all_labels[num_of_training_samples:].reshape(-1,img_size, img_size,1)


'''
extract validation data
'''

validation_samples = int(validation_split * imgs.shape[0])
validation_imgs = imgs[-validation_samples:].reshape(-1,img_size, img_size,1)
validation_labels = labels[-validation_samples:].reshape(-1,img_size,img_size,1)

imgs = imgs[:-validation_samples].reshape((-1,img_size,img_size,1))
labels = labels[:-validation_samples].reshape(-1,img_size, img_size,1)


  
inputs = imgs.reshape((-1, img_size, img_size,1))
val_im = validation_imgs.reshape((-1,img_size,img_size,1))
val_lab = validation_labels

print("output shape", labels.shape)

'''
####################################################################
                    DATA AUGMENTATION
####################################################################
'''

training_data, training_labels = augment_data(imgs, labels, epochs = args.g_epochs) 

'''
####################################################################
                    TRAINING
####################################################################
'''



print("\n  Pre - Training : \n ")
  
 
model.fit(training_data, training_labels, validation_data = (val_im, val_lab), epochs = args.pretrain_epochs, batch_size = batch_size)


critic = networks.get_critic_scan(args.c_filters, lr = 2e-4)
critic.summary()
for l in critic.layers:
    l.trainable = False
stacked_inp = Input((img_size, img_size,1))
stacked_unet = model(stacked_inp)
stacked_critic = critic(stacked_unet)
stacked_model = Model(inputs = stacked_inp, outputs = stacked_critic)
stacked_model.compile(optimizer = Adam(args.stacked_lr), loss='binary_crossentropy')

for l in critic.layers:
    l.trainable = True

n_batches = int(training_data.shape[0]/batch_size)

y_real = np.ones((batch_size,1))
y_fake = np.zeros((batch_size,1))

dices = []
val_dices = []

for epoch in range(args.epochs):
  progress_bar = Progbar(target=n_batches)
  print("\nepoch " , epoch, " of ", args.epochs)
  dice_avg = 0.
  
  for ind_batch in range(int(n_batches)):

    progress_bar.update(ind_batch)
    scans_batch = training_data[batch_size * ind_batch:(ind_batch + 1) * batch_size,:,:]

    labels_batch = training_labels[batch_size * ind_batch:(ind_batch + 1) * batch_size]
    for i in range(3):
      model_loss = model.train_on_batch(scans_batch, labels_batch)
    
    pred_labels = model.predict(scans_batch)
    c_weights = critic.layers[0].get_weights()  
    for i in range(args.critic_epochs):
      c2 = critic.train_on_batch(labels_batch, y_real)
      c1 = critic.train_on_batch(pred_labels, y_fake)

      
    c_loss = 0.5*(c1 + c2)  
    stacked_loss = stacked_model.train_on_batch(scans_batch, y_real)
    progress_bar.add(1, values=[("dice", model_loss[0]), ("critic loss", c_loss), ("fooling loss", stacked_loss)])
    dice_avg += model_loss[1]
    
  val_dice = model.evaluate(val_im, val_lab)[1]
  val_fooling = stacked_model.evaluate(val_im , np.ones(val_im.shape[0]), verbose = 0)
  val_samples = model.predict(val_im, verbose = 0)
  val_critic_loss = critic.evaluate(np.concatenate([val_samples, val_lab]), np.concatenate([np.zeros(val_samples.shape[0]), np.ones(val_lab.shape[0]) ]) ,verbose = 0)
  progress_bar.add(1, values=[("val_dice", val_dice), ("critic loss", val_critic_loss), ("fooling loss", val_fooling)])
  
  dices.append(dice_avg/n_batches)
  val_dices.append(val_dice)

  
'''
####################################################################
                    PLOTTING AND TESTING
####################################################################
'''
  
  
fig, ax = plt.subplots(1,1)

ax.plot(dices, label = "$DICE_{train}$")
ax.plot(val_dices, label="$DICE_{val}$")
ax.legend(loc = 'lower right')
ax.set_xlabel("Epoche")
ax.set_ylim([0.5,1.])
fig.savefig("./../figures/hist" + str(args.stacked_lr) + str(args.n_filters) + ".png")

  
'''
evaluation on test data:
'''

print("\n   Testing: \n")
tic = time.time()
predictions = model.predict(test_imgs.reshape(-1,256,256,1))

toc = time.time()
prediction_time = toc-tic
print("predicting took " ,prediction_time , "seconds")
np.save("./../predictions/predictions"+ str(args.stacked_lr) + str(args.n_filters) +".npy", predictions)
DICE = util.test_evaluation(test_labels, predictions,filename=args.res_file)

plt.show()