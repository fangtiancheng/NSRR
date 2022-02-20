from turtle import forward
import torch
import torch.nn as nn
import torchvision
import pytorch_ssim

#from pytorch_msssim import ssim
from utils import SingletonPattern
from model import LayerOutputModelDecorator

from typing import List


def feature_reconstruction_loss(conv_layer_output: torch.Tensor, conv_layer_target: torch.Tensor) -> torch.Tensor:
    """
    Computes Feature Reconstruction Loss as defined in Johnson et al. (2016)
    todo: syntax
    Justin Johnson, Alexandre Alahi, and Li Fei-Fei. 2016. Perceptual losses for real-time
    style transfer and super-resolution. In European Conference on Computer Vision.
    694–711.
    Takes the already-computed output from the VGG16 convolution layers.
    """
    # print("conv_layer_output.shape = ", conv_layer_output.shape)
    # print("conv_layer_target.shape = ", conv_layer_target.shape)
    if conv_layer_output.shape != conv_layer_target.shape:
        raise ValueError("Output and target tensors have different dimensions!")
    loss = conv_layer_output.dist(conv_layer_target, p=2) / torch.numel(conv_layer_output)
    return loss

class NSRRLoss(nn.Module):
    def __init__(self, w: float, device: torch.device) -> None:
        super(NSRRLoss, self).__init__()
        self.w = w
        self.device = device
        self.perceptualLossManager = PerceptualLossManager(device)
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss as defined in the NSRR paper.
        """
        B, C, H, W = output.shape
        tot_loss = 0.
        for i in range(B):
            output_i = output[i, :, :, :].unsqueeze(0)
            target_i = target[i, :, :, :].unsqueeze(0)
            loss_ssim = 1 - pytorch_ssim.ssim(output_i, target_i)
            loss_perception = 0
            conv_layers_output = self.perceptualLossManager.get_vgg16_conv_layers_output(output_i)
            conv_layers_target = self.perceptualLossManager.get_vgg16_conv_layers_output(target_i)
            for i in range(len(conv_layers_output)):
                loss_perception += feature_reconstruction_loss(conv_layers_output[i], conv_layers_target[i])
            tot_loss += loss_ssim + self.w * loss_perception
        return tot_loss / B
        
class MSELoss(torch.nn.Module):
    def __init__(self, *params) -> None:
        super(MSELoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss()
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return self.loss_fn(output, target)

class PerceptualLossManager(metaclass=SingletonPattern):
    """
    Singleton
    """
    # Init
    def __init__(self, device):
        self.vgg_model = torchvision.models.vgg16(pretrained=True, progress=True).to(device)
        self.vgg_model.eval()
        """ 
            Feature Reconstruction Loss 
            - needs output from each convolution layer.
        """
        self.layer_predicate = lambda name, module: type(module) == nn.Conv2d
        self.lom = LayerOutputModelDecorator(self.vgg_model.features, self.layer_predicate)

    def get_vgg16_conv_layers_output(self, x: torch.Tensor)-> List[torch.Tensor]:
        """
        Returns the list of output of x on the pre-trained VGG16 model for each convolution layer.
        """
        return self.lom.forward(x)

