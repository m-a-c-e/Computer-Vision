"""
Utilities to be used along with the deep model
"""
from typing import Union

import torch
from vision.my_resnet import MyResNet18
from vision.simple_net import SimpleNet
from vision.simple_net_final import SimpleNetFinal
from vision.multilabel_resnet import MultilabelResNet18
from torch import nn


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy given the prediction logits and the ground-truth labels

    Args:
        logits: The output of the forward pass through the model.
                for K classes logits[k] (where 0 <= k < K) corresponds to the
                log-odds of class `k` being the correct one.
                Shape: (batch_size, num_classes)
        labels: The ground truth label for each instance in the batch
                Shape: (batch_size)
    Returns:
        accuracy: The accuracy of the predicted logits
                   (number of correct predictions / total number of examples)
    """
    batch_accuracy = 0.0
    ############################################################################
    # Student code begin
    ############################################################################

    predicted_labels    = torch.argmax(logits, dim=1)
    correct_predictions = torch.sum(predicted_labels == labels)    
    batch_accuracy      = (correct_predictions.item() / logits.shape[0])

    ############################################################################
    # Student code end
    ############################################################################
    return batch_accuracy

def compute_loss(
    model: Union[SimpleNet, SimpleNetFinal, MyResNet18, MultilabelResNet18],
    model_output: torch.Tensor,
    target_labels: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    """
    Computes the loss between the model output and the target labels

    Args:
    -   model: a model (which inherits from nn.Module)
    -   model_output: the raw scores output by the net
    -   target_labels: the ground truth class labels
    -   is_normalize: bool flag indicating that loss should be divided by the batch size
    Returns:
    -   the loss value
    """
    loss = None

    ############################################################################
    # Student code begin
    ############################################################################

    batch_size = model_output.size()[0]
    loss = model.loss_criterion(model_output, target_labels) 
    if is_normalize:
        loss = loss / batch_size 

    ############################################################################
    # Student code end
    ############################################################################

    return loss

def compute_multilabel_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy given the prediction logits and the ground-truth labels

    Args:
        logits: The output of the forward pass through the model.
                for K labels logits[k] (where 0 <= k < K) corresponds to the
                log-odds of label `k` being present in the image.
                Shape: (batch_size, num_labels)
        labels: The ground truth label for each instance in the batch
                Shape: (batch_size, num_labels)
    Returns:
        accuracy: The accuracy of the predicted logits
                  (number of correct predictions / total number of labels)
    """
    batch_accuracy = 0.0
    ############################################################################
    # Student code begin
    ############################################################################
    logits = logits.to('cpu')
    labels = labels.to('cpu')
    batch_size           = logits.shape[0]
    num_class_labels     = logits.shape[1]
    logits         = torch.where(logits > 0.5, torch.tensor(1), torch.tensor(0))
    compare_output = torch.where(labels == logits, torch.tensor(1), torch.tensor(0)) 
    batch_accuracy = torch.sum(compare_output).item() / (batch_size * num_class_labels)

    ############################################################################
    # Student code end
    ############################################################################

    return batch_accuracy


def save_trained_model_weights(
    model: Union[SimpleNet, SimpleNetFinal, MyResNet18, MultilabelResNet18], out_dir: str
) -> None:
    """Saves the weights of a trained model along with class name

    Args:
    -   model: The model to be saved
    -   out_dir: The path to the folder to store the save file in
    """
    class_name = model.__class__.__name__
    state_dict = model.state_dict()

    assert class_name in set(
        ["SimpleNet", "SimpleNetFinal", "MyResNet18", "MultilabelResNet18"]
    ), "Please save only supported models"

    save_dict = {"class_name": class_name, "state_dict": state_dict}
    torch.save(save_dict, f"{out_dir}/trained_{class_name}_final.pt")
