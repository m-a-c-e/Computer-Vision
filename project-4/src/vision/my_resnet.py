from pyexpat import model
import torch
import torch.nn as nn
from torchvision.models import resnet18
#import torchvision.models as models


class MyResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

        ############################################################################
        # Student code begin
        ############################################################################
    
        model = resnet18(pretrained=True)
    
        for param in model.parameters():
            param.requires_grad = False

        # mod_list = [mod for mod in model.children()]
        # all_layers = mod_list[0:-1]
        
        self.conv_layers = nn.Sequential(*(list(model.children())[:-1]))
        self.fc_layers = nn.Linear(512, 15)

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape,  (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        ############################################################################
        # Student code begin
        ############################################################################

        x = self.conv_layers(x).reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        model_output = x 

        ############################################################################
        # Student code end
        ############################################################################
        return model_output