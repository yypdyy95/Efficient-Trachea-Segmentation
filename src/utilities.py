import numpy as np

import matplotlib.pyplot as plt



def filter_anisotropic_diffusion(I, n_iter = 2, kappa = 8e-4, gamma = 1e-4):
    '''
    anisotropic diffusion Filter
        computes anisotroic diffusion for given number of timesteps
        using kappa * exp(-|grad(I)|^2)

    arguments:
        I       - input image
        niter   - number of steps diffusion is applied
        kappa   - controls anisotropy (weighted with exp(-x^2/kappa))
        gamma   - diffusion parameter -> controls speed of diffusion

    returns:
        filtered - filtered input image

      Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.
    '''
      
    I = I / np.max(I)
    
    filtered = I.copy()
    
    x_grad = np.zeros_like(I)
    y_grad = x_grad.copy()
    laplacian_x = x_grad.copy()
    laplacian_y = x_grad.copy()

    for i in range(n_iter):
        #compute gradients
        x_grad[:,:-1] = np.diff(filtered, 0,axis = 0)[:,:-1]
        y_grad[:-1,:] = np.diff(filtered, 0,axis = 1)[:-1,:]
        
        anisotropy_x = np.exp(- x_grad**2 / kappa)
        anisotropy_y = np.exp(- y_grad**2 / kappa)

        # discretised diffusion equation: (for 1D)
        # dI/dt = (I(x+1) - 2* I(x) + I(x-1))/delta_x^2, delta_x ==1
        # subtract shifted differences for second derivatices
        # np.diff gives x[n] - x[n-1] -> laplacian = - (diff(x) - diff())

        laplacian_x[:,1:] = - x_grad[:,1:] + x_grad[:,:-1]
        laplacian_y[1:,:] = - y_grad[1:,:] + y_grad[:-1,:]

        filtered += gamma * (anisotropy_x * laplacian_x + anisotropy_y *laplacian_y)
        filtered[:, -1] = filtered[:,-2]
        filtered[-1,:] = filtered[-2,:]

    filtered = filtered * 255.0
    filtered = filtered.astype(np.uint8)
    return filtered



def get_model_name(name, filtersize, number_of_filters, dropout, batch_norm, cropping):

    model_name = name + "_" + str(filtersize) + "_" + str(number_of_filters)
    if dropout:
      model_name = model_name + "_D"
    if batch_norm:
      model_name = model_name + "_B"
    if cropping:
      model_name = model_name + "_C"

    return model_name


def test_evaluation(labels, segmentations, filename = 'unet_results.txt', plot = False, print_all = False):
    '''
    test_evaluation:
      - evaluation of segmentation, writes
    arguments:
      labels : numpy array - GT data
      segmentations: numpy array - predicted segmentations
      filename: string - file to write results to
      plot: bool - if results should be plottet
    returns:
      dice: test dice score
    '''

    x_dim = segmentations.shape[2]

    segmentations_orig = segmentations
    segmentations = segmentations > 0.5

    segmentations = segmentations.astype(np.uint8)

    difs  = labels - segmentations
    sums   = labels + segmentations
    num_of_samples = int(labels.shape[0]/40)
    dices = np.zeros(num_of_samples)
    tps = np.zeros(num_of_samples)
    fns = np.zeros(num_of_samples)
    fps = np.zeros(num_of_samples)

    for i in range(num_of_samples):

      su  = sums[i* 40 : (i+1)*40]
      dif = difs[i* 40 : (i+1)*40]
      area = np.sum(labels[i* 40 : (i+1)*40])
      TP = np.sum(su  == 2)
      FN = np.sum(dif == 1)
      FP = np.sum(dif == -1)

      DICE = 2* TP/(2*TP + FP + FN)
      dices[i] = DICE
      fps[i] = FP/area
      fns[i] = FN/area
      tps[i] = TP/area

    print("\n ################ summary: #####################\n")

    print("DICE \t= {:.3f}\t+- {:.3f}".format(np.sum(dices)/num_of_samples, np.std(dices)))
    print("TP \t= {:.3f}\t+- {:.3f}".format(np.sum(tps)/num_of_samples, np.std(tps)))
    print("FN \t= {:.3f}\t+- {:.3f}".format(np.sum(fns)/num_of_samples, np.std(fns)))
    print("FP \t= {:.3f}\t+- {:.3f}".format(np.sum(fps)/num_of_samples, np.std(fps)))

    with open(filename, 'a') as res_file:
        
      res_file.write("DICE \t= " + "{:.3f}".format(np.sum(dices/num_of_samples)) + "\t+-\t" + "{:.3f}".format(np.std(dices)) + "\n")
      res_file.write("TP \t= " + "{:.3f}".format(np.sum(tps)/num_of_samples) + "\t+-\t" +"{:.3f}".format(np.std(tps)) + "\n")
      res_file.write("FN \t= " + "{:.3f}".format(np.sum(fns)/num_of_samples) + "\t+-\t" + "{:.3f}".format(np.std(fns)) + "\n")
      res_file.write("FP \t= " + "{:.3f}".format(np.sum(fps)/num_of_samples) + "\t+-\t" +  "{:.3f}".format(np.std(fps))+ "\n")

    TP = np.sum(sums  == 2)
    FN = np.sum(difs == 1)
    FP = np.sum(difs == -1) 

    if plot:
      fig, ax = plt.subplots(1,3)
      fig.set_size_inches(13.6, 7)

      for i in range(400):

        ax[0].cla()
        ax[1].cla()
        ax[2].cla()

        ax[0].imshow(segmentations[i], vmin = 0, vmax = 1)
        ax[2].imshow(segmentations_orig[i], cmap = "Greys_r", vmin = 0, vmax = 1)
        ax[1].imshow(labels[i]- segmentations[i], vmin = -1,vmax = 1)
        ax[0].set_title("slice "+ str(i) )

        if (np.mod(i,40) < 5):
          plt.pause(0.5)
        else:
           plt.pause(0.1)
      plt.show()
      
    return np.sum(dices/num_of_samples)

    

def test_evaluation_slices(labels, segmentations, filename = 'unet_results.png', plot = True, print_all = False):
    '''
    test_evaluation:
      - evaluation of segmentation, writes
    arguments:
      labels : numpy array - GT data
      segmentations: numpy array - predicted segmentations
      filename: string - file to write results to
      plot: bool - if results should be plottet
    returns:
      dice: test dice score
    '''

    x_dim = segmentations.shape[2]

    segmentations_orig = segmentations
    segmentations = segmentations > 0.5

    segmentations = segmentations.astype(np.uint8)

    difs = labels - segmentations
    sums = labels + segmentations
    n_samples = int(labels.shape[0]/40)
    dices = np.zeros((n_samples, 40))
    tps = np.zeros((n_samples, 40))
    fns = np.zeros((n_samples, 40))
    fps = np.zeros((n_samples, 40))

    for i in range(40):
      for slice in range(n_samples):
        index = slice * 40 + i
        area = np.sum(labels[index]) + 1e-5
        
        TP = np.sum(sums[index]  == 2)
        FN = np.sum(difs[index] == 1)
        FP = np.sum(difs[index] == -1)
        
        DICE = (2* TP + 1e-5)/(2*TP + FP + FN + 1e-5)
        
        dices[slice, i] = DICE
        fps[slice, i] = FP
        fns[slice, i] = FN
        tps[slice, i] = TP
    
    dice_avg = np.zeros(40)
    fp_avg = np.zeros(40)
    tp_avg = np.zeros(40)
    fn_avg = np.zeros(40)  
    
    dice_std = np.zeros(40)
    fp_std = np.zeros(40)
    tp_std = np.zeros(40)
    fn_std = np.zeros(40)  
    
    for i in range(40):
      dice_avg[i] = np.sum(dices[:,i])/n_samples
      fp_avg[i] = np.sum(fps[:,i])/n_samples
      fn_avg[i] = np.sum(fns[:,i])/n_samples
      tp_avg[i] = np.sum(tps[:,i])/n_samples
      
      dice_std[i] = np.std(dices[:,i])
      fp_std[i] = np.std(fps[:,i])
      fn_std[i] = np.std(fns[:,i])
      tp_std[i] = np.std(tps[:,i])
      
  
    if plot:
      fig, ax = plt.subplots(2,2)
      ax = ax.reshape(-1)
      fig.set_size_inches(8, 8)
      x = range(1,41)
      fmt = 's'
      capsize = 2.
      markerfacecolor = 'white'
      markersize = 3
      color = '#ef11ff'
      ax[0].errorbar(x = x,y = dice_avg, yerr=dice_std, fmt = fmt, markerfacecolor = markerfacecolor, markersize = markersize, color = color, capsize = capsize)
      ax[0].set_ylim(0.5,1.1)
      ax[0].set_xlabel("Ebene")
      ax[0].set_ylabel("$DICE_{2D}$-Koeffizient")
      
      ax[1].errorbar(x = x,y = fp_avg, yerr=fp_std, fmt = fmt, markerfacecolor = markerfacecolor, markersize = markersize, color = color, capsize = capsize)
      ax[1].set_xlabel("Ebene")
      ax[1].set_ylabel("False Positives")
      
      ax[2].errorbar(x = x,y = fn_avg, yerr=fn_std, fmt = fmt, markerfacecolor = markerfacecolor, markersize = markersize, color = color, capsize = capsize)
      ax[2].set_xlabel("Ebene")
      ax[2].set_ylabel("False Negatives")
      
      ax[3].errorbar(x = x,y = tp_avg, yerr=tp_std, fmt = fmt, markerfacecolor = markerfacecolor, markersize = markersize, color = color, capsize = capsize)
      ax[3].set_xlabel("Ebene")
      ax[3].set_ylabel("True Positives")
    
      plt.show()
      fig.savefig(filename)

      
def get_data(n_samples = 4400):
    '''
    get_data:
      - loads stored training and test data as numpy arrays
      arguments:
      - n_samples: 
        - Number of samples to load
        - defualt = 4400 -> load all samples 
        - usefull for testing new features, saving time to load 
      returns:
      - imgs, labels, test_imgs, test_labels:
          tuple of numpy arrays, data divided in train and test set
        
    '''
    
    imgs = np.load("./../data/scans/training_data_n.npy").reshape((-1,256,256))[:n_samples]
    labels = np.load("./../data/labels/training_labels_n.npy").reshape((-1,256,256))[:n_samples]

    test_imgs = np.load("./../data/scans/test_data_n.npy").reshape((-1,256,256))[:n_samples]
    test_labels = np.load("./../data/labels/test_labels_n.npy").reshape((-1,256,256))[:n_samples]
    return imgs, labels, test_imgs, test_labels