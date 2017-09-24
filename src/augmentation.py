from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def elastic_transformation2D(I, sigma = (None, None), alpha = None):
    '''
    elastic_transformation2D
      - used for data augmentation
      
      - applies small random distortions to a greyscale image, single pixels will be translated randomly. Results in slight stretching or squeezing in some areas.
      
      - initialize random field, apply gaussian filter to it, shift pixel indicees according to that field -> map new coordinates  
      
    arguments:
      - I: numpy array - image to be transformed
      - sigma: tuple of floats - standard deviations for gaussian filter
      - alpha: magnitude of transformations
    
    '''

    # initialize random state
    I = I[:,:,0]

    if sigma == (None, None):
      sigma = (30 * np.random.rand(), 30* np.random.rand())
    if alpha == None:
      alpha = 1e-3
    random_state = np.random.RandomState(None)
    img_shape = I.shape
    # deviations in x, y, and z directions:
    # rand gives values in range [0,1] -> normalize them to [-1,1]
    dx = alpha * gaussian_filter(random_state.rand(*img_shape) * 2.0 - 1 ,  sigma, mode = 'constant', cval = 0.0)
    dy = alpha * gaussian_filter(random_state.rand(*img_shape) * 2.0 - 1 ,  sigma, mode = 'constant', cval = 0.0)
    # now create index array and shift indicees according to random field


    x,y = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]), indexing = 'ij')
    indices = (np.reshape(x+dx,(-1,1)), np.reshape(y+dy,(-1,1)) )

    return map_coordinates(I, indices, order=1).reshape(256,256,1).astype(np.uint8)



def augment_data(imgs, labels, epochs = 1):
  
  
  '''
  augment data:
  - generating new training samples using keras image data generator and elastic transformations
  arguments: 
   - imgs: images to augment
   - labels: labels to augment
   - epochs: number of augmentations to be done 
  returns:
   - augm_imgs, augm_labels: tuple of numpy arrays, concatenation of original an augmented data
  '''
    
  datagen = ImageDataGenerator(
      samplewise_center=False,
      samplewise_std_normalization=False,
          rotation_range=1,
          width_shift_range=0.001,
          height_shift_range=0.1,
          shear_range=0.01,
          zoom_range=0.01,
          fill_mode='reflect',
      preprocessing_function = elastic_transformation2D
      )

  gen_ep = epochs
  augm_imgs = imgs
  augm_labels = labels
  
  '''
  because one has to augment images and labels seperately taking a seed, such that same transformations will be performed on both 
  '''
  
  seed = np.random.randint(256)
  for x_batch, y_batch in datagen.flow(imgs, labels, batch_size=imgs.shape[0], seed = seed):
    gen_ep -= 1
    if gen_ep < 0:
      break
    augm_imgs = np.concatenate([augm_imgs, x_batch])
    
  gen_ep = epochs
  for y_batch, x_batch in datagen.flow(labels, imgs, batch_size=imgs.shape[0], seed = seed):
    gen_ep -= 1
    if gen_ep < 0:
      break
    y_batch = (y_batch > 0).astype(np.uint8)
    augm_labels = np.concatenate([augm_labels, y_batch])


  return augm_imgs, augm_labels
  