from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import warnings

from torch.cuda.amp import GradScaler
import faiss
warnings.filterwarnings('ignore')

from utils.AverageMeter import *
from utils.other_utils import *
from utils.utils_mixup import *
from utils.losses import *

import wandb


def train_algo(args, scheduler, model,  device, 
              train_loader, train_selected_loader, optimizer, epoch):
    
    train_loss = AverageMeter()

    model.train()
    end = time.time()
    counter = 1

    criterionCE = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing)
    scaler = GradScaler()

    for batch_idx, (img, labels, index) in enumerate(train_selected_loader):

        model.zero_grad()
        img1, img2, labels = img[0].to(device), img[1].to(device), labels.to(device)

        if args.mixup:
            img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, args.alpha_m, device)
            img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, args.alpha_m, device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds1, _ = model(img1)
                preds2, _ = model(img2)

                if args.cr:
                    loss = ClassificationLoss(args, preds1, preds2, y_a1, y_b1, y_a2, y_b2,
                                        lam1, lam2, criterionCE, epoch, device)
                else:
                    loss = ClassificationLoss2(args, preds2, y_a2, y_b2, 
                                               lam2, criterionCE, epoch, device)          
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds1, _ = model(img1)
                preds2, _ = model(img2)

            if args.cr:
                loss = ClassificationLoss4(args, preds1, preds2, labels, criterionCE, epoch, device)
            else:
                loss = ClassficationLoss3(args, preds2, labels, criterionCE, epoch, device)
    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss.update(loss.item(), img1.size(0))        
          
        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(img1), len(train_loader.dataset),
                       100. * counter / len(train_loader), 0,
                optimizer.param_groups[0]['lr']))
        counter = counter + 1
    print('train_class_loss',train_loss.avg)
    print('train time', time.time()-end)



def reliable_pseudolabel_selection(args, device, trainloader, features, epoch, model_preds=None):
    
    features_numpy = features.cpu().numpy() 
    index = faiss.IndexFlatIP(features_numpy.shape[1])
    index.add(features_numpy)
    partial_labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))
    labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))
    clean_labels = torch.LongTensor(trainloader.dataset.clean_labels)
    soft_labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))

    D,I = index.search(features_numpy, args.k_val+1)
    neighbors = torch.LongTensor(I)
    weights = torch.exp(torch.Tensor(D[:,0:])/0.1)
    N = features_numpy.shape[0]

    if epoch > args.start_correct:
        prob, pred = torch.max(model_preds,1)
        prob, pred = prob.squeeze(), pred.squeeze()      
        
        conf_th = args.conf_th_h - (args.conf_th_h - args.conf_th_l) * ((epoch - args.start_correct)/(args.epoch - args.start_correct))
        conf_id = (prob > conf_th).nonzero().reshape(-1)
        print('Confident model predictions:', len(conf_id))
        print('Correct confident model predictions:',(pred[conf_id] == clean_labels[conf_id]).sum())
        print('Model pred already in partial set:',(partial_labels[conf_id, pred[conf_id]] == 1).sum())

        soft_labels[conf_id, pred[conf_id]] = 1
        labels[conf_id, pred[conf_id]] = 1


    print('New Clean pl:',(labels[range(N),clean_labels] == 1).sum())
    wandb.log({'Clean pl':(labels[range(N),clean_labels] == 1).sum()},epoch)

    score = torch.zeros(N, args.num_classes)

    knn_indices = neighbors.view(N, args.k_val+1, 1).expand(N, args.k_val+1, args.num_classes)
    knn_soft_labels = soft_labels.expand(N, -1, -1)

    score = torch.sum(
        torch.mul(
            torch.gather(knn_soft_labels, 1, knn_indices),
            weights.view(N, -1, 1),
        ),  # batch_size x k x feature_dim
        1,
    )

    pseudo_labels = torch.max(score, -1)[1]
    soft_labels = torch.zeros((len(pseudo_labels),args.num_classes)).scatter_(1, pseudo_labels.view(-1,1), 1)

    correct_soft_labels = (pseudo_labels == clean_labels).sum()
    match_id = (partial_labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct pseudo labels:', correct_soft_labels)
    print('Correct pseudo label matches:', correct_soft_label_matches, match_id.int().sum())


    wandb.log({
        'Correct pseudo labels': correct_soft_labels,
        'Correct pseudo label matches': correct_soft_label_matches,
        'Total noisy_pseudo matches': match_id.int().sum()
    }, epoch)


    match_id = (labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct pseudo label matches with  pred+partial:', correct_soft_label_matches, match_id.int().sum())

    wandb.log({
        'Correct pseudo label matches with  pred+partial': correct_soft_label_matches,
    }, epoch)


    knn_soft_labels = soft_labels.expand(N, -1, -1)

    score = torch.sum(
        torch.mul(
            torch.gather(knn_soft_labels, 1, knn_indices),
            weights.view(N, -1, 1),
        ),  # batch_size x k x feature_dim
        1,
    ) 


    pseudo_labels = torch.max(score, -1)[1]
    soft_labels = score/score.sum(1).unsqueeze(-1)

    correct_soft_labels = (pseudo_labels == clean_labels).sum()
    match_id = (partial_labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()


    # correct_soft_labels = ((torch.max(soft_labels, dim=1)[1])[clean_id] == clean_labels[clean_id]).sum()
    # match_id = (torch.max(soft_labels, dim=1)[1] == labels) # only one for every image
    # correct_soft_label_matches = (labels[match_id] == clean_labels[match_id]).int().sum()

    print('Correct posterior labels:', correct_soft_labels)
    print('Correct posterior label matches:', correct_soft_label_matches)

    wandb.log({
        'Correct posterior labels': correct_soft_labels,
        'Correct posterior label matches': correct_soft_label_matches
    }, epoch)

    match_id = (labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct posterior label matches with  pred+partial:', correct_soft_label_matches, match_id.int().sum())

    wandb.log({
        'Correct posterior label matches with  pred+partial': correct_soft_label_matches,
    }, epoch)


    prob_temp = soft_labels[:,:].clone()
    prob_temp[prob_temp<=1e-2] = 1e-2
    prob_temp[prob_temp>(1-1e-2)] = 1-1e-2
    discrepancy_measure2 = -torch.log(prob_temp)

    agreement_measure = torch.zeros((N, args.num_classes)).float()
    agreement_measure[range(N),torch.max(soft_labels, dim=1)[1]] = (labels[range(N),torch.max(soft_labels, dim=1)[1]] == 1.0).float().data.cpu()
        
    print('Init Matches 2:', (agreement_measure[range(N), clean_labels] == 1.0).int().sum())

    num_clean_per_class = torch.zeros(args.num_classes)
    for i in range(args.num_classes):
        num_clean_per_class[i] = torch.sum(agreement_measure[:,i])

    if(args.delta==0.5):
        num_samples2select_class = torch.median(num_clean_per_class)
    elif(args.delta==1.0):
        num_samples2select_class = torch.max(num_clean_per_class)
    elif(args.delta==0.0):
        num_samples2select_class = torch.min(num_clean_per_class)
    else:
        num_samples2select_class = torch.quantile(num_clean_per_class,args.delta)

    print(num_clean_per_class)
    print(num_samples2select_class)

    agreement_measure = torch.zeros((len(labels),))
    selected_examples_labels = torch.zeros((len(clean_labels),args.num_classes))+float('inf')


    for i in range(args.num_classes):
        idx_class = labels[:,i] == 1.0
        samples_per_class = idx_class.sum()
        idx_class = (idx_class.float()==1.0).nonzero().squeeze()
        discrepancy_class = discrepancy_measure2[idx_class, i]

        k_corrected = min(num_samples2select_class, samples_per_class)
        val, top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)
        agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0
        selected_examples_labels[idx_class[top_clean_class_relative_idx],i] = val

    _,selected_labels = torch.min(selected_examples_labels,1)

    selected_examples = agreement_measure
    print('selected examples', sum(selected_examples))

    correct_selected_examples = (selected_labels[selected_examples.bool()] == clean_labels[selected_examples.bool()]).int().sum()
    print('Correct Selected examples:',correct_selected_examples)

    wandb.log({
        'Selected examples': sum(selected_examples),
        'Correct selected examples': correct_selected_examples,
    }, epoch)

    return selected_examples, selected_labels




def reliable_pseudolabel_selection_weighted(args, device, trainloader, features, epoch, model_preds=None):
    
    features_numpy = features.cpu().numpy() 
    index = faiss.IndexFlatIP(features_numpy.shape[1])
    index.add(features_numpy)
    partial_labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))
    labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))
    clean_labels = torch.LongTensor(trainloader.dataset.clean_labels)
    soft_labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))
    prior = torch.Tensor(np.copy(trainloader.dataset.weights))


    D,I = index.search(features_numpy, args.k_val+1)
    neighbors = torch.LongTensor(I)
    weights = torch.exp(torch.Tensor(D[:,0:])/0.1)
    N = features_numpy.shape[0]

    if epoch > args.start_correct:
        prob, pred = torch.max(model_preds,1)
        prob, pred = prob.squeeze(), pred.squeeze()      
        
        conf_th = args.conf_th_h - (args.conf_th_h - args.conf_th_l) * ((epoch - args.start_correct)/(args.epoch - args.start_correct))
        conf_id = (prob > conf_th).nonzero().reshape(-1)
        print('Confident model predictions:', len(conf_id))
        print('Correct confident model predictions:',(pred[conf_id] == clean_labels[conf_id]).sum())
        print('Model pred already in partial set:',(partial_labels[conf_id, pred[conf_id]] == 1).sum())

        soft_labels[conf_id, pred[conf_id]] = 1
        labels[conf_id, pred[conf_id]] = 1

    soft_labels_p = torch.mul(soft_labels, prior)

    print('New Clean pl:',(labels[range(N),clean_labels] == 1).sum())
    wandb.log({'Clean pl':(labels[range(N),clean_labels] == 1).sum()},epoch)

    score = torch.zeros(N, args.num_classes)

    knn_indices = neighbors.view(N, args.k_val+1, 1).expand(N, args.k_val+1, args.num_classes)
    knn_soft_labels = soft_labels_p.expand(N, -1, -1)

    score = torch.sum(
        torch.mul(
            torch.gather(knn_soft_labels, 1, knn_indices),
            weights.view(N, -1, 1),
        ),  # batch_size x k x feature_dim
        1,
    )

    pseudo_labels = torch.max(score, -1)[1]
    soft_labels = torch.zeros((len(pseudo_labels),args.num_classes)).scatter_(1, pseudo_labels.view(-1,1), 1)

    correct_soft_labels = (pseudo_labels == clean_labels).sum()
    match_id = (partial_labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct pseudo labels:', correct_soft_labels)
    print('Correct pseudo label matches:', correct_soft_label_matches, match_id.int().sum())


    wandb.log({
        'Correct pseudo labels': correct_soft_labels,
        'Correct pseudo label matches': correct_soft_label_matches,
        'Total noisy_pseudo matches': match_id.int().sum()
    }, epoch)


    match_id = (labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct pseudo label matches with  pred+partial:', correct_soft_label_matches, match_id.int().sum())

    wandb.log({
        'Correct pseudo label matches with  pred+partial': correct_soft_label_matches,
    }, epoch)


    knn_soft_labels = soft_labels.expand(N, -1, -1)

    score = torch.sum(
        torch.mul(
            torch.gather(knn_soft_labels, 1, knn_indices),
            weights.view(N, -1, 1),
        ),  # batch_size x k x feature_dim
        1,
    ) 


    pseudo_labels = torch.max(score, -1)[1]
    soft_labels = score/score.sum(1).unsqueeze(-1)

    correct_soft_labels = (pseudo_labels == clean_labels).sum()
    match_id = (partial_labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()


    # correct_soft_labels = ((torch.max(soft_labels, dim=1)[1])[clean_id] == clean_labels[clean_id]).sum()
    # match_id = (torch.max(soft_labels, dim=1)[1] == labels) # only one for every image
    # correct_soft_label_matches = (labels[match_id] == clean_labels[match_id]).int().sum()

    print('Correct posterior labels:', correct_soft_labels)
    print('Correct posterior label matches:', correct_soft_label_matches)

    wandb.log({
        'Correct posterior labels': correct_soft_labels,
        'Correct posterior label matches': correct_soft_label_matches
    }, epoch)

    match_id = (labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct posterior label matches with  pred+partial:', correct_soft_label_matches, match_id.int().sum())

    wandb.log({
        'Correct posterior label matches with  pred+partial': correct_soft_label_matches,
    }, epoch)


    prob_temp = soft_labels[:,:].clone()
    prob_temp[prob_temp<=1e-2] = 1e-2
    prob_temp[prob_temp>(1-1e-2)] = 1-1e-2
    discrepancy_measure2 = -torch.log(prob_temp)

    agreement_measure = torch.zeros((N, args.num_classes)).float()
    agreement_measure[range(N),torch.max(soft_labels, dim=1)[1]] = (labels[range(N),torch.max(soft_labels, dim=1)[1]] == 1.0).float().data.cpu()
        
    print('Init Matches 2:', (agreement_measure[range(N), clean_labels] == 1.0).int().sum())

    num_clean_per_class = torch.zeros(args.num_classes)
    for i in range(args.num_classes):
        num_clean_per_class[i] = torch.sum(agreement_measure[:,i])

    if(args.delta==0.5):
        num_samples2select_class = torch.median(num_clean_per_class)
    elif(args.delta==1.0):
        num_samples2select_class = torch.max(num_clean_per_class)
    elif(args.delta==0.0):
        num_samples2select_class = torch.min(num_clean_per_class)
    else:
        num_samples2select_class = torch.quantile(num_clean_per_class,args.delta)

    print(num_clean_per_class)
    print(num_samples2select_class)

    agreement_measure = torch.zeros((len(labels),))
    selected_examples_labels = torch.zeros((len(clean_labels),args.num_classes))+float('inf')


    for i in range(args.num_classes):
        idx_class = labels[:,i] == 1.0
        samples_per_class = idx_class.sum()
        idx_class = (idx_class.float()==1.0).nonzero().squeeze()
        discrepancy_class = discrepancy_measure2[idx_class, i]

        k_corrected = min(num_samples2select_class, samples_per_class)
        val, top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)
        agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0
        selected_examples_labels[idx_class[top_clean_class_relative_idx],i] = val

    _,selected_labels = torch.min(selected_examples_labels,1)

    selected_examples = agreement_measure
    print('selected examples', sum(selected_examples))

    correct_selected_examples = (selected_labels[selected_examples.bool()] == clean_labels[selected_examples.bool()]).int().sum()
    print('Correct Selected examples:',correct_selected_examples)

    wandb.log({
        'Selected examples': sum(selected_examples),
        'Correct selected examples': correct_selected_examples,
    }, epoch)

    return selected_examples, selected_labels