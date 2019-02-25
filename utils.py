import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def parseData(args, sample, split='train'):
    input, target, mask = sample['img'], sample['N'], sample['mask'] 
    if args.cuda:
        input  = input.cuda(); target = target.cuda(); mask = mask.cuda(); 

    input_var  = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    mask_var   = torch.autograd.Variable(mask, requires_grad=False);

    data = {'input': input_var, 'tar': target_var, 'm': mask_var}
    if args.in_light:
        light = sample['light'].expand_as(input)
        if args.cuda: light = light.cuda()
        light_var = torch.autograd.Variable(light);
        data['l'] = light_var
    return data 
   
def errorPred(true, pred, mask=None):
    # Tensor is of dimension n x c x h x w
    
    true_n = f.normalize(true, p=2, dim=1)
    pred_n = f.normalize(pred, p=2, dim=1)
    #dot_product = (true * pred).sum(1).clamp(-1,1)
    inner_product = (true_n * pred_n).sum(1)
    error_map = torch.acos(inner_product) * 180.0 / math.pi 
    error_map = error_map * mask.narrow(1, 0, 1).squeeze(1)

    err_den = mask.narrow(1, 0, 1).sum()
    err_num = error_map[mask.narrow(1, 0, 1).squeeze(1).byte()]
    mean_err = err_num.sum() / err_den
    return mean_err

class Criterion(object):
    def __init__(self, args):
        self.loss_fn = torch.nn.CosineEmbeddingLoss()
        if args.cuda:
            self.loss_fn = self.loss_fn.cuda()

    def forward(self, out_normal, true_normal):
        num = true_normal.nelement()/true_normal.shape[1]
        if not hasattr(self, 'flag') or num != self.flag.nelement():
            self.flag = torch.autograd.Variable(true_normal.data.new().resize_(num).fill_(1))

        self.out_reshape = out_normal.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        self.true_reshape = true_normal.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        self.loss = self.loss_fn(self.out_reshape, self.true_reshape, self.flag)
        out_loss = self.loss.item()
        return out_loss

    def backward(self):
        self.loss.backward()
        
class Criterion_mask(object):
    def __init__(self, args):
        pass

    def forward(self, out_normal, true_normal):
        num = true_normal.nelement()/true_normal.shape[1]
        if not hasattr(self, 'flag') or num != self.flag.nelement():
            self.flag = torch.autograd.Variable(true_normal.data.new().resize_(num).fill_(1))
        n, c, h, w = out_normal.shape
        out_reshape = out_normal.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        true_reshape = true_normal.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        loss_pixel = 1 - f.cosine_similarity(out_reshape, true_reshape, dim=1)
        print(loss_pixel.mean())
        loss_pixel = loss_pixel / loss_pixel.max()
        mask = torch.bernoulli(loss_pixel)
        mask = mask.detach()
        #mask = torch.where(loss_pixel >= threshold, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
        loss_final = (loss_pixel*mask).mean() / mask.mean()
        loss_final = loss_final.cuda()
        self.loss = loss_final
        out_loss = self.loss.item()
        return out_loss

    def backward(self):
        self.loss.backward()