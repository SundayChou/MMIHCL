"""
Functions for calculating losses in models.
"""
import torch
import torch.nn.functional as fc


def contrastLoss(embeds_list_1, embeds_list_2, n_layers=2, temp=0.1):
    """
    Calculate the contrastive loss for embeddings.

    Parameters:
    embeds_list_1 (list): The first list of embeddings on each layer.
    embeds_list_2 (list): The second list of embeddings on each layer.
    n_layers (int): The number of layers.
    temp (float): The temperature parameter.

    Returns:
    loss (torch.Tensor): The contrastive loss.
    """
    loss = 0

    for i in range(1, n_layers + 1):
        embeds_1 = embeds_list_1[i].detach()
        embeds_2 = embeds_list_2[i]

        loss += contrastLossOneLayer(embeds_1, embeds_2, temp)

    return loss


def contrastLossOneLayer(embeds_1, embeds_2, temp=0.1):
    """
    Calculate the contrastive loss for embeddings of the same layer.

    Parameters:
    embeds_1 (torch.Tensor): The first group of embeddings.
    embeds_2 (torch.Tensor): The second group of embeddings.
    temp (float): The temperature parameter.

    Returns:
    loss (torch.Tensor): The contrastive loss of the same layer.
    """
    embeds_1 = fc.normalize(embeds_1 + 1e-7, p=2)
    embeds_2 = fc.normalize(embeds_2 + 1e-7, p=2)

    nume = torch.exp(torch.sum(embeds_1 * embeds_2, dim=-1) / temp)
    deno = torch.exp(embeds_1 @ embeds_2.T / temp).sum(-1) + 1e-7

    loss = -torch.log(nume / deno).mean()

    return loss


def L2Norm(x):
    """
    L2 normalize a tensor.

    Parameters:
    x (torch.Tensor): The input tensor.

    Returns:
    y (torch.Tensor): The L2 normalized tensor.
    """
    eps = torch.FloatTensor([1e-7]).cuda()
    y = x / (torch.max(torch.norm(x, dim=1, keepdim=True), eps))

    return y