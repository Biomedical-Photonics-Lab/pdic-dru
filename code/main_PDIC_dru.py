import os.path
import glob
import cv2
import logging
import time
import sys

import numpy as np
from datetime import datetime
from collections import OrderedDict
import hdf5storage

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import utils_model
from utils import utils_pnp as pnp
from utils import utils_sisr as sr
from utils import utils_image as util
import scipy.io as sio
from adm2tvl2 import ADM2TVL2_step, structtype, cNorm

#sys.path.append('/home/minxu/research/ML/CNNIQA')
#from test_demo import *

#import matplotlib.pyplot as plt
from PIL import Image
import matlab.engine
eng = matlab.engine.start_matlab()  
eng.addpath(r"code/Matlab"); # add path to the Matlab folder on git

import scipy.io



###############################
#
# Tasks: Upgrade activation function to tanh.
#
###############################



"""
Spyder (Python 3.7)
PyTorch 1.6.0
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; homepage: https://cszn.github.io/)
by Kai Zhang (01/August/2020)

# --------------------------------------------
|--model_zoo               # model_zoo
   |--drunet_gray          # model_name, for color images
   |--drunet_color
|--testset                 # testsets
|--results                 # results
# --------------------------------------------
"""



def KK(im, opts):
    [m, n] = im.shape[:2]
    # This must be odd
    my2 = np.floor(m/2); my = 2*my2+1
    mx2 = np.floor(n/2); mx = 2*mx2+1
    #my2 = (m-1)/2; my = 2*my2+1
    #mx2 = (n-1)/2; mx = 2*mx2+1

    ky = np.arange(-mx2,mx2+1)/(2*mx2 + 1)
    kx = np.arange(-my2,my2+1)/(2*my2 + 1)

    kx = np.repeat(kx[...,None], 2*mx2+1, axis=1)
    ky = np.repeat(ky[None,...], 2*my2+1, axis=0)

    k = np.sqrt(kx*kx + ky*ky);

    ## Shift them to the regular Fourier pattern and add the 2*pi factor
    kx = 2*np.pi*np.fft.ifftshift(kx)
    ky = 2*np.pi*np.fft.ifftshift(ky)
    k = 2*np.pi*np.fft.ifftshift(k)

    if 1:
        ## Incorporate PSF for the objective. It is important to use more accurate psf
        # w = 0.42*opts.wavelength/opts.NA/opts.pixelsz          # width for psf
        # psf = np.exp(-1/2*w*w*k*k)
        kmat = matlab.single((k/opts.pixelsz).tolist())
        kmat.reshape(k.shape)
        psf = eng.CBF(kmat, opts.NAc, opts.NAo, opts.wavelength)
        psf = psf*np.sinc(kx/(2*np.pi)*opts.shear/opts.pixelsz)
      
        ## PDIC only measures \partial phi/\partial x and hence only needs to modify kx
        kx = kx*(0.9*psf + 0.1)

    return kx, ky
    

def fourierSol(Kx, Ky, F, eps1=1e-10, eps2=0.01):
    FB = -1j*Kx*F
    X = FB.div(Kx*Kx + eps2*Ky*Ky + eps1)
    return torch.real(torch.fft.ifftn(X, dim=(-2, -1)))
   

def show(end, pos, x, skip=0, flag=0):
    im = np.squeeze(np.array(x.cpu()))
    if skip > 0:
        im = im[skip:-skip, skip:-skip]
    im = Image.fromarray(im)
    image_mat = matlab.single(list(im.getdata()))
    image_mat.reshape((im.size[0], im.size[1]))
    eng.subplot(pos)
    eng.imagesc(image_mat)
    eng.axis('square', nargout=0)
    eng.colorbar()
    #eng.colormap('gray')
    score = 0
    if flag:
        ## ML/sr-metric: *** quality_prediction is slow yet most accurate! ***
        ## score = eng.quality_predict(image_mat)
        ## ML/GSVD: gsvd is in-accurate!
        #score = mage_eng.gsvd(imat)
        print('>>> Image: {:.5f} {:.5f} {:.5g} <- Quality'.format(x.max().cpu(), x.min().cpu(), score))
        return score
    else:
       print('>>> Image: {:.5f} {:.5f}'.format(x.max().cpu(), x.min().cpu()))

## This is also not accurate!
## ML/CNNIQA
def qual(y, device):
    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)

    model.load_state_dict(torch.load("/home/minxu/research/ML/CNNIQA/models/CNNIQA-LIVE"))

    im = Image.fromarray(y.cpu().detach().numpy()).convert('L')
    
    patches = NonOverlappingCropPatches(im, 32, 32)

    model.eval()
    with torch.no_grad():
        patch_scores = model(torch.stack(patches).to(device))

    return patch_scores.mean().item()


def main():

    """
    # ----------------------------------------------------------------------------------
    # In real applications, you should set proper 
    # - "noise_level_img": from [3, 25], set 3 for clean image, try 15 for very noisy LR images
    # - "k" (or "kernel_width"): blur kernel is very important!!!  kernel_width from [0.6, 3.0]
    # to get the best performance.
    # ----------------------------------------------------------------------------------
    """
    ##############################################################################

    testset_name = 'PDIC2'                # set test set,  'set5' | 'srbsd68'
    x8 = True                            # default: False, x8 to boost performance
    model_name = 'drunet_tanh'           # 'ircnn_color'         # set denoiser, | 'drunet_color' | 'ircnn_gray' | 'drunet_gray' | 'ircnn_color' | 'drunet_tanh'
    sf = 1                               # set scale factor, 1, 2, 3, 4
    iter_num = 41                       # set number of iterations, default: 24 for SISR
    noise_level_img = 3                  # set noise level of image, from [3, 25], set 3 for clean image
    noise_level_model = noise_level_img/255.  # noise level of model
    
    """
    # set your own kernel width !!!!!!!!!!
    """
    kernel_width = 1.0


    ##############################################################################
    opts = structtype()                # For 60x water immersion lens 
    opts.shear = 0.22                  # micron
    opts.pixelsz = 0.0564              # 60x objective
    opts.NA = 0.55 + 1.2
    opts.NAo = 1.2
    opts.NAc = 0.55
    opts.wavelength = 0.545

    
    ##############################################################################
    task_current = 'sr'                  # 'sr' for super-resolution
    n_channels = 1 if 'gray' in model_name or 'tanh' in model_name else 3  # fixed
    act_mode = 'H' if 'tanh' in model_name else 'R'
    model_zoo = 'model_zoo'              # fixed
    testsets = 'testsets'                # fixed
    results = 'results '                  # fixed
    result_name = testset_name + '_realapplications_' + task_current + '_' + model_name
    model_path = os.path.join(model_zoo, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------
    with_drunet = True
    if 'drunet' in model_name:
        from models.network_unet import UNetRes as net
        model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode=act_mode, downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):
        #if idx < 5:
        #    continue
        
        # --------------------------------
        # (1) get img_L
        # --------------------------------
        logger.info('Model path: {:s} Image: {:s}'.format(model_path, img))
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = np.float32(sio.loadmat(img)['Delta'])
        if np.ndim(img_L)==2:
            #img_L = img_L[0:511, 0:511, None]
            img_L = img_L[..., None]


        if 'x40' in img or 'X40' in img or '40x' in img or '40X' in img:
            opts.pixelsz = 0.0893
            opts.shear = 0.34
            opts.NAo = 0.60
            opts.NAc = 0.55
            opts.NA = opts.NAo + opts.NAc
            
        if 'Siemens' in img:
            opts.pixelsz = np.float32(np.squeeze(sio.loadmat(img)['opts']['pixelsz']))
            ref = np.float32(sio.loadmat(img)['oplOverEtch'])
            #ref = ref[0:511, 0:511, None]
            ref = ref[..., None]

        # 1/sigma2 = 1/sigma^2 is the SNR
        I = np.float32(sio.loadmat(img)['I'])
        gamma = 0.176                         # For the camera quantum efficiency
        sigma2 = (opts.pixelsz/opts.shear)**2*0.5/gamma/np.mean(I) # variance of noise in phi_x from intensity measurement
        
        # --------------------------------
        # (2) get rhos and sigmas: weighting between the fidelity and prior terms
        # --------------------------------
        mu = 1/sigma2
        
        ## TV-part: alpha varies between 1 and 20.
        ## 00_Siemens: alpha=1; 01_sphere: alpha=20.  
        if "Siemens" in img:
            alpha = 1           # For simulated Siemens and Siemens_fine
        else:
            alpha = 20          # For measured samples with x60 PDIC
        beta = alpha*20 #mu/100 #200 #100 #*alpha #np.logspace(np.log10(1), np.log10(mu/100), iter_num).astype(np.float32)
        gamma1 = 1.618

        ## Adaptive iteration
        eta = 0.9
        early_termination = False
        
        # --------------------------------
        # (3) initialize x, and pre-calculation
        # --------------------------------
        # kx, Ky
        Kx, Ky = KK(img_L, opts)

        if np.ndim(Kx)==2:
            Kx = Kx[..., None]
            Ky = Ky[..., None]
        Kx = util.single2tensor4(Kx).to(device, dtype=torch.float)
        Ky = util.single2tensor4(Ky).to(device, dtype=torch.float)

        # FT{im}
        b = img_L
        b = util.single2tensor4(b).to(device, dtype=torch.float)
        b = b - b.mean()
        b = torch.nn.functional.pad(b, (0, Kx.shape[-2]-b.shape[-2], 0, Kx.shape[-1]-b.shape[-1]))
        if 'Siemens' in img:
            ref = util.single2tensor4(ref).to(device, dtype=torch.float)
            ref = torch.nn.functional.pad(ref, (0, Kx.shape[-2]-ref.shape[-2], 0, Kx.shape[-1]-ref.shape[-1]))
        F = torch.fft.fftn(b, dim=(-2, -1))

        ## Fourier solution
        if 1:
            xf = fourierSol(Kx, Ky, F, eps1=1e-6, eps2=0.01)
            sio.savemat(os.path.join(E_path, img_name+'_x'+str(sf)+'_'+model_name+'_fourier.mat'), {'S': xf.squeeze().cpu().detach().numpy()})
            fidelity = (1j*Kx*torch.fft.fftn(xf, dim=(-2,-1)) - F).norm()/np.sqrt(xf.shape[-1]*xf.shape[-2])/(xf.shape[-1]*xf.shape[-2])
            fidelity1 = (1j*Kx*torch.fft.fftn(xf, dim=(-2,-1)) - F).norm(p=1)/np.sqrt(xf.shape[-1]*xf.shape[-2])/(xf.shape[-1]*xf.shape[-2])
            print('    Fourier solution fidelity: {} {}'.format(fidelity, fidelity1))
            if 'Siemens' in img:
                print('    Difference: {}'.format(cNorm(xf,ref)/ref.norm()))
            show(eng, '121', xf, flag=1)


        # initialize x, w, z, lam
        x = np.zeros(Kx.shape)
        w = np.zeros(Kx.shape)
        lam = np.zeros(Kx.shape)
        x = torch.from_numpy(x).to(device, dtype=torch.float)
        w = torch.from_numpy(w).to(device, dtype=torch.float)
        lam = torch.from_numpy(lam).to(device, dtype=torch.float)

        # Obtain sigma from the Fourier solutions 
        if 1:
            s, s2, s4, N = 0, 0, 0, 500
            for ind in range(N):
                F1 = F + torch.fft.fftn(torch.randn(x.shape, device=device)*np.sqrt(sigma2), dim=(-2,-1))
                x1 = fourierSol(Kx, Ky, F1, eps1=1e-6, eps2=0.01)
                s = s + x1 
                s2 = s2 + x1*x1
                s4 = s4 + x1*x1*x1*x1
            s, s2, s4 = s/N, s2/N, s4/N
            varu = (s2 - s*s).mean()
            n4 = (s4 - s*s*s*s - 6*s*s*(s2 - s*s)).mean()
            varvaru = n4/N - varu*varu*(N-3)/N/(N-1)
            print('   var(u), std(var(u))/var(u), var(u)/sigma2: {} {} {}'.format(varu, torch.sqrt(varvaru)/varu, varu/sigma2))

            sigma = torch.sqrt(varu).cpu()
            # For   01_sphere: var(u)/sigma2 = 17.1*(1 +- 0.06)
            #      03_stained: var(u)/sigma2 = 15.6*(1 +- 0.07)
            #    04_unstained: var(u)/sigma2 = 15.6*(1 +- 0.07)
            #      00_Siemens: var(u)/sigma2 = 9.9*(1 +- 0.06)
            # 05_Siemens_fine: var(u)/sigma2 = 7.7*(1 +- 0.07)

        # denoiser-part
        if any(str in img for str in ['stained', 'Stained', 'PR243', 'Pr243', 'PR632', 'Pr632', 'sphere']):
            xi0 = 0.1           # for stained and unstained tissue
        elif 'sphere' in img: 
            xi0 = 0.8           # for spheres
        elif 'Siemens' in img: 
            xi0 = 0.8           # for simulated Siemens target
        else:
            raise Exception("xi0 should be defined.")
        
        modelSigma1 = xi0 # max(1000/np.sqrt(mu), xi0) # set sigma_1, default: 49 out of 255
        modelSigma2 = sigma                      # the uncertainty in the recovered phase
        xi = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32) # add another iteration with same xi 
        xi21 = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), 41).astype(np.float32)
        gamma = 0.23

        if 0:
            sigma2 = 1/0.176/np.mean(I)
            mu = 1/(0.23*sigma2)
            alpha = 1
            beta = mu/100
            modelSigma1 = 49/255*np.sqrt(0.23)
            modelSigma2 = np.sqrt(sigma2*0.23)
            gamma1 = 1
            xi = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)

        print('   sigma, sqrt(sigma2), ratio: {} {} {}'.format(sigma, np.sqrt(sigma2), sigma/np.sqrt(sigma2)))

        mu, alpha, beta, xi, gamma1, sigma2, xi0, gamma = torch.tensor(mu).to(device, dtype=torch.float), torch.tensor(alpha).to(device, dtype=torch.float), torch.tensor(beta).to(device, dtype=torch.float), torch.tensor(xi).to(device, dtype=torch.float), torch.tensor(gamma1).to(device, dtype=torch.float), torch.tensor(sigma2).to(device, dtype=torch.float), torch.tensor(xi0).to(device, dtype=torch.float), torch.tensor(gamma).to(device, dtype=torch.float)
 
        
        # --------------------------------
        # (4) main iterations
        # --------------------------------
        delta, deltap, deltapp, residue, residuep = 1, 1, 1, 1, 1
        for i in range(iter_num):

            success = False

            while not success:
                print('\nIter: {} / {} {} {}'.format(i, iter_num, mu, xi[i]))

                # --------------------------------
                # step 1, FFT
                # --------------------------------
                tau = (1/(xi[i]*xi[i])).float().repeat(1, 1, 1, 1)
                xp, wp, lamp = x, w, lam
                if with_drunet:
                    x, w, lam = ADM2TVL2_step(x, w, lam, F, Kx, Ky, mu, tau, alpha=alpha, beta=beta, gamma=gamma1) #tvpart+
                else:
                    tau[0,0,0,0] = 0
                    x, w, lam = ADM2TVL2_step(x, w, lam, F, Kx, Ky, mu, tau, alpha=alpha, beta=beta, gamma=gamma1, eps=1e-6)

        
                # x: [0, 0, x direction, y direction]
                pri_residue = (x.diff(axis=-1, append=x[...,0][...,None]) - w).norm()
                ds = w - wp
                dual_residue = beta*ds.diff(axis=-1, prepend=ds[...,-1][...,None]).norm()
                pri_thres = max(x.diff(axis=-1, append=x[...,0][...,None]).norm(), w.norm())
                dual_thres = lam.diff(axis=-1, prepend=lam[...,-1][...,None]).norm()
                residue, residuep = (pri_residue/pri_thres + dual_residue/dual_thres)/2, residue
                
                fidelity = (1j*Kx*torch.fft.fftn(x, dim=(-2,-1)) - F).norm()/np.sqrt(x.shape[-1]*x.shape[-2])/(x.shape[-1]*x.shape[-2])
                fidelity1 = (1j*Kx*torch.fft.fftn(x, dim=(-2,-1)) - F).norm(p=1)/np.sqrt(x.shape[-1]*x.shape[-2])/(x.shape[-1]*x.shape[-2])

                if pri_residue/pri_thres > 2*dual_residue/dual_thres:
                    beta = 2*beta
                elif pri_residue/pri_thres < 0.5*dual_residue/dual_thres:
                    beta = beta/2

                print('    Fidelity, w, relative residue primary, dual: {} {} {} {} {}'.format(fidelity, fidelity1, w.norm(p=1)/(x.shape[-1]*x.shape[-2]), pri_residue/pri_thres, dual_residue/dual_thres))

                if 'Siemens' in img:
                    print('    Difference: {} {}'.format((x-xp).norm()/x.norm(), cNorm(x,ref)/ref.norm()))
                else:
                    print('    Difference: {}'.format((x-xp).norm()/x.norm()))

                show(eng, '121', x)
                
                
                # --------------------------------
                # step 2, denoiser
                # --------------------------------
                if with_drunet:
                    if x8:
                        x = util.augment_img_tensor4(x, i % 8)

                    if 'drunet' in model_name and True: 
                        x = torch.cat((x, torch.min(xi[i], xi0)*torch.sqrt(gamma).repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
                        x = utils_model.test_mode(model, x, mode=2, refield=64, min_size=256, modulo=16)
                    
                    if x8:
                        if i % 8 == 3 or i % 8 == 5:
                            x = util.augment_img_tensor4(x, 8 - i % 8)
                        else:
                            x = util.augment_img_tensor4(x, i % 8)

                    show(eng, '122', x)

                delta, deltap, deltapp = (x-xp).norm()/x.norm(), delta, deltap
                print('==> delta: {} {} {}'.format(delta, deltap, deltapp))
                
                
                if 1:
                    if delta > eta*deltap and deltap != 1.0:
                        success = True
                    else:
                        success = False
                else:
                    success = True

                if delta > deltap and delta > deltapp and (residue < 0.15 or residuep < 0.15) and deltap < 0.005:
                   x = xp
                   early_termination = True

            if early_termination or not with_drunet:
                break

        score = show(eng, '121', x, flag=1)
                
        # --------------------------------
        # (3) img_E
        # --------------------------------
        sio.savemat(os.path.join(E_path, img_name+'_x'+str(sf)+'_'+model_name+'.mat'), {'S': x.squeeze().cpu().detach().numpy(), 'alpha': alpha.cpu().numpy(), 'beta': beta.cpu().numpy(), 'mu': mu.cpu().numpy(), 'xi': xi.cpu().numpy(), 'xi0': xi0.cpu().numpy(), 'gamma': gamma.cpu().numpy(), 'gamma1': gamma1.cpu().numpy(), 'score': score})
        
        img_E = util.tensor2uint(x)
        util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(sf)+'_'+model_name+'.png'))

        #break

if __name__ == '__main__':

    main()
