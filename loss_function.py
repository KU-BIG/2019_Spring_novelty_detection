import torch
from torch.nn import functional as F

def loss_function(recon_x, x, mu, logvar, BATCH_SIZE, input_dim) :
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    KLD /= BATCH_SIZE * input_dim
    
    return BCE + KLD