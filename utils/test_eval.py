from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

def accuracy_v3(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    result = []
    maxk = max(top)
    batch_size = preds.size(0)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t() # pred[k-1] stores the k-th predicted label for all samples in the batch.
    correct = pred.eq(labels.view(1,-1).expand_as(pred))

    for k in top:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k)

    return result


def accuracy_v2(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    result = []
    maxk = max(top)
    batch_size = preds.size(0)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t() # pred[k-1] stores the k-th predicted label for all samples in the batch.
    correct = pred.eq(labels.view(1,-1).expand_as(pred))

    for k in top:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))

    return result



def test_eval(args, model, device, test_loader, epoch=0):
    model.eval()
    loss_per_batch = []
    test_loss = 0
    correct_1 = 0
    correct_5 = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            try:
                output, _ = model(data)
            except:
                output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            #pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            result = accuracy_v3(output, target, top=[1,5])
            correct_1 += result[0].item()
            correct_5 += result[1].item()
            #correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx == 0:
                model_preds = output.cpu()
                gt = target.cpu()
            else:
                model_preds = torch.cat((model_preds, output.cpu()), 0)
                gt = torch.cat((gt, target.cpu()),0)

    test_loss /= len(test_loader.dataset)
    print('\nTest set prediction branch: Average loss: {:.4f}, top1 Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct_1, len(test_loader.dataset),
        100. * correct_1 / len(test_loader.dataset)))
    print('\nTest set prediction branch: Average loss: {:.4f}, top5 Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct_5, len(test_loader.dataset),
        100. * correct_5 / len(test_loader.dataset)))
    
    acc = [0 for c in range(args.num_classes)]
    for c in range(args.num_classes):
        acc[c] = ((torch.max(model_preds,-1)[1] == gt) * (gt == c)).sum() / (gt == c).sum()
    print('Avg Accuracy:', sum(acc)/args.num_classes, acc)

    wandb.log({
            'Avg Accuracy': sum(acc)/args.num_classes
        }, epoch)

        
    loss_per_epoch = np.average(loss_per_batch)
    acc_val_per_epoch = np.array(100. * correct_1 / len(test_loader.dataset))

    return (loss_per_epoch, acc_val_per_epoch)