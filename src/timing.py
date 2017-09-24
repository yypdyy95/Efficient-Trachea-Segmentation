from networks import *
import utilities as util 
import time
import numpy as np
from skimage.exposure import equalize_hist

imgs, labels, test_imgs, test_labels = util.get_data(40)

start_time = time.time()
print(imgs.shape)
for img in imgs:
    img = util.filter_anisotropic_diffusion(img, n_iter = 1, gamma = 0.01, kappa=1)

print("time for diffusion filter {}".format(time.time() - start_time))

t1 = time.time()    
for img in imgs:    
    img = equalize_hist(img)
print("time for HE {}".format(time.time() - t1))

t2 = time.time()
imgs = imgs.reshape(-1,256*256)

for img in imgs:  
    img /= np.std(img)+1e-5  
    img -= np.mean(img)    
imgs = imgs.reshape(-1,256,256,1)
print("time for rescaling: {}".format(time.time() - t2 ))

print("overall time for preprocessing: {}".format(time.time() - start_time))  

model, model_name, out_dim = get_unet(filtersize = 3, n_filters = 4, dropout = False, batch_norm = True)
model.predict(imgs)
t3 = time.time()
model.predict(imgs)
print("prediction time: {}".format(time.time() - t3))