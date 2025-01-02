import torch
from PIL import Image

from loss import SR_l
from loss.pytorch_ssim import *
def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    return gradient_h, gradient_w

def tv_loss(illumination):
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    loss_h = gradient_illu_h
    loss_w = gradient_illu_w
    loss = loss_h.mean() + loss_w.mean()
    return loss

def SR_g_loss(R1, R2):
    loss = torch.nn.MSELoss()(R1, R2) 
    return loss

def SR_l_loss(R1, R2):
    L_spa = SR_l.L_spa()
    loss = torch.mean(L_spa(R1, R2))
    return loss

def RB_loss(L1, R1, im1, X1):
    max_rgb1, _ = torch.max(im1, 1)
    max_rgb1 = max_rgb1.unsqueeze(1) 
    log_L1 = torch.log(L1)
    log_R1 = torch.log(R1)
    log_X1 = torch.log(X1)
    #loss1 = torch.nn.MSELoss()(L1*R1, X1) + torch.nn.MSELoss()(R1, X1/L1.detach())
    loss1 = torch.nn.MSELoss()(log_L1+log_R1, log_X1) + torch.nn.MSELoss()(log_R1, log_X1-log_L1.detach())
    loss2 = torch.nn.MSELoss()(L1, max_rgb1) + tv_loss(L1)
    return loss1 + loss2

def PD_loss(im1, X1):
    loss = torch.nn.MSELoss()(im1, X1)
    return loss

def joint_RGB_horizontal(im1, im2):
    if im1.size==im2.size:
        w, h = im1.size
        result = Image.new('RGB',(w*2, h))
        result.paste(im1, box=(0,0))
        result.paste(im2, box=(w,0))      
    return result

def joint_L_horizontal(im1, im2):
    if im1.size==im2.size:
        w, h = im1.size
        result = Image.new('L',(w*2, h))
        result.paste(im1, box=(0,0))
        result.paste(im2, box=(w,0))   
    return result
    
def ssim_loss(EI, im):
    ssim_loss = SSIM()
    loss = 1 - ssim_loss(EI, im)
    return loss