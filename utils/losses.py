import torch
import torch.nn.functional as F

def ClassificationLoss(args, preds1, preds2, y_a1, y_b1, y_a2, y_b2, 
                       lam1, lam2, criterionCE, epoch, device):
    """ CR and Mix-up"""
    preds = torch.cat((preds1, preds2), dim=0)

    targets_a = torch.cat((y_a1, y_a2), dim=0)
    targets_b = torch.cat((y_b1, y_b2), dim=0)

    ones_vec = torch.ones((preds1.size(0),)).float().to(device)
    lam_vec = torch.cat((lam1 * ones_vec, lam2 * ones_vec), dim=0).to(device)

    loss = lam_vec * criterionCE(preds, targets_a) + (1 - lam_vec) * criterionCE(preds, targets_b)
    loss = loss.mean()
    return loss


def ClassificationLoss2(args, preds1, y_a1, y_b1, 
                       lam1, criterionCE, epoch, device):
    """ No CR and Mix-up """

    ones_vec = torch.ones((preds1.size(0),)).float().to(device)
    lam_vec = (lam1 * ones_vec).to(device)

    loss = lam_vec * criterionCE(preds1, y_a1) + (1 - lam_vec) * criterionCE(preds1, y_b1)
    loss = loss.mean()
    return loss

def ClassficationLoss3(args, preds1, y1, criterionCE, epoch, device):
    """ No CR and No Mix-up """
    loss = criterionCE(preds1, y1)
    loss = loss.mean()
    return loss

def ClassificationLoss4(args, preds1, preds2, y, 
                       criterionCE, epoch, device):
    """CR and No Mix-up """
    preds = torch.cat((preds1, preds2), dim=0)

    targets = torch.cat((y, y), dim=0)

    loss = criterionCE(preds, targets)
    loss = loss.mean()
    return loss


