import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time

from data.crowdsource import *
from utils.other_utils import *
from utils.features import *
from utils.lr_scheduler import *
from utils.utils_algo import *
from utils.test_eval import *
from utils.cutout import *
from utils.autoaugment import *
from models.preact_resnet import *
from models.pretrained_resnet import *
from models.wideresnet import *
from models.densenet import *
from models.resnext import *
from models.senet import *

from torch import optim
from torchvision import transforms
import random
import sys

import wandb
wandb.login()

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--epoch', type=int, default=200, help='training epoches')
    parser.add_argument('--start_correct', type=int, default=1, help='partial label augmentation')  
    parser.add_argument('--lr', '--base-learning-rate', '--base-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--lr-warmup-epoch', type=int, default=1, help='warmup epoch')
    parser.add_argument('--lr-warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[60, 120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
                      
    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of in-distribution classes')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--noise_type', default='partial', help='partial or none')
    parser.add_argument('--noise_ratio', type=float, default=0.3, help='ratio of noise')
    parser.add_argument('--partial_ratio', type=float, default=0.05, help='ratio of partial noise')
    parser.add_argument('--heirarchical', type=boolean_string, default=False, help='heirarchical noise CIFAR100')
    parser.add_argument('--train_root', default='./dataset', help='root for train data')
    parser.add_argument('--out', type=str, default='./out', help='Directory of the output')
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='name of the experiment (for the output files)')
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='CIFAR-10, CIFAR-100, CUB-200')
    parser.add_argument('--download', type=boolean_string, default=False, help='download dataset')

    parser.add_argument('--network', type=str, default='R18', help='Network architecture')
    parser.add_argument('--seed_initialization', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int, default=42, help='random seed (default: 1)')
    
    parser.add_argument('--alpha_m', type=float, default=1.0, help='Beta distribution parameter for mixup')
    parser.add_argument('--delta', type=float, default=0.5, help='example selection th')
    parser.add_argument('--k_val', type=int, default=250, help='k for knn')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing')
    # parser.add_argument('--smooth_k', type=int, default=1, help='smooth_k')
    parser.add_argument('--conf_th_h', type=float, default=1.0, help='confidence threshold high')
    parser.add_argument('--conf_th_l', type=float, default=1.0, help='confidence threshold low')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers')
    parser.add_argument('--num_workers_sel', type=int, default=4, help='num workers sel')
    parser.add_argument('--cr', type=boolean_string, default=True, help='consistency regularization')
    parser.add_argument('--mixup', type=boolean_string, default=True, help='Mixup')
    parser.add_argument('--slice', type=int, default=1, help='data slices')
    parser.add_argument('--lpi', type=int, default=10, help='data slices')

    args = parser.parse_args()
    return args



def data_config(args, transform_train_w, transform_train_s, transform_test):

    trainset, testset = get_dataset(args, TwoTransform(transform_train_w, transform_train_s), transform_test)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('############# Data loaded #############')

    return train_loader, test_loader, trainset

def build_model(args,device, path=None):

    if args.network == 'R18' or args.network == 'R50':
        if args.dataset == 'CUB-200' or args.dataset == 'Treeversity' or args.dataset == 'Benthic' or args.dataset == 'Plankton':
            model = PretrainedResNet(num_classes=args.num_classes,arch=args.network).to(device)
        else:
            model = ResNet18(num_classes=args.num_classes).to(device)

    elif args.network == 'resnext':
        model = resnext50(num_classes=args.num_classes).to(device)
    elif args.network == 'senet':
        model = seresnet50(num_classes=args.num_classes).to(device)
    elif args.network == 'wideresnet':
        model = WideResNet(34, args.num_classes, widen_factor=10, dropRate=0.0).to(device)
    elif args.network == 'densenet':
        model = densenet121(num_classes=args.num_classes).to(device)
    
    if path:
        model.load_state_dict(torch.load(path),strict=False)

    return model




def main(args=None):

    wandb.init(project='PALS', config=args, mode='disabled')
    # args = wandb.config
  
    exp_path = os.path.join(args.out, 'noise_models_' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name, args.seed_initialization, args.seed_dataset), args.noise_type, str(args.partial_ratio),str(args.noise_ratio),str(args.heirarchical))
    res_path = os.path.join(args.out, 'metrics' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name, args.seed_initialization, args.seed_dataset), args.noise_type, str(args.partial_ratio),str(args.noise_ratio),str(args.heirarchical))
    print('Output Folder:', res_path)

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
                            
    __console__=sys.stdout
    name= "/results"
    log_file=open(res_path+name+".log",'a')
    sys.stdout=log_file
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed_initialization)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed

    random.seed(args.seed_initialization)  # python seed for image transformation

    if args.dataset == 'CIFAR-10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'CIFAR-100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif args.dataset == 'CUB-200':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    elif args.dataset == 'Treeversity':
        mean = [0.4439581940620345, 0.4509297096690951, 0.3691211738638277]
        std = [0.23407518616927706, 0.22764417468550843, 0.2600833107790479]
    elif args.dataset == 'Benthic':
        mean = [0.34728872821176615, 0.40013687864974884, 0.4110478166769647]
        std = [0.1286915489786319, 0.13644626747739305, 0.14258506692263767]
    elif args.dataset == 'Plankton':
        mean = [0.9663359216202008, 0.9663359216202008, 0.9663359216202008]
        std = [0.10069729102981237, 0.10069729102981237, 0.10069729102981237]



    if args.dataset == 'CUB-200' or args.dataset=='Treeversity':
        transform_train_w = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
        transform_train_s = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif args.dataset == 'Benthic':
        transform_train_w = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
        transform_train_s = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((112,112)),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif args.dataset == 'Plankton':
        transform_train_w = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((96,96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
        transform_train_s = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_train_w = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        transform_train_s = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std)
        ])

    if args.dataset == 'CUB-200' or args.dataset=='Treeversity':
        transform_test = transforms.Compose(
        [
        transforms.Resize(int(224/0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    elif args.dataset == 'Benthic':
        transform_test = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif args.dataset == 'Plankton':
        transform_test = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


    train_loader, test_loader, trainset = data_config(args, transform_train_w, transform_train_s, transform_test)

    model = build_model(args,device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    # scheduler = get_scheduler(optimizer, len(train_loader), args)

    optimizer = torch.optim.SGD([
                    {'params': model.encoder.parameters(), 'lr': args.lr / 100},          # volcanic-sweep-10 parameters.
                    {'params': model.fc.parameters(), 'lr': args.lr},
                ], momentum=args.momentum, weight_decay=args.wd)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    for epoch in range(1, args.epoch + 1):
        st = time.time()
        print("=================>    ", args.experiment_name,args.partial_ratio, args.noise_ratio)

        features, model_preds = compute_features(args,model,train_loader,transform_test, device, epoch, ret_pred=True)
        if epoch==1:
            print ('Features Shape',features.shape )
        if(epoch<=args.start_correct):        
            print('######## Pseudo-labelling ########')
            selected_examples, selected_examples_labels = reliable_pseudolabel_selection_weighted(args, device, train_loader, features, epoch)
        elif(epoch > args.start_correct):
            print('######## Pseudo-labelling ########')
            selected_examples, selected_examples_labels = reliable_pseudolabel_selection_weighted(args, device, train_loader, features, epoch, model_preds)
        
        trainset.targets = selected_examples_labels
        train_loader.dataset.targets = selected_examples_labels
        train_selected_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers_sel, pin_memory=True, 
                                                            sampler=torch.utils.data.WeightedRandomSampler(selected_examples, len(selected_examples)))
        train_algo(args, scheduler,model,device, train_loader, train_selected_loader, optimizer, epoch)
        
        print('Epoch time: {:.2f} seconds\n'.format(time.time()-st))
        log_file.flush()
        print('######## Test ########')
        
        loss_per_epoch, acc_val_per_epoch = test_eval(args, model, device, test_loader, epoch=epoch)

        wandb.log({
            'Test Loss': loss_per_epoch,
            'Test Accuracy': acc_val_per_epoch
        }, epoch)

        
        if (epoch %10 ==0):
            save_model(model, optimizer, args, epoch, exp_path+"/model.pth")
            # np.save(res_path + '/' + 'selected_examples_train.npy', selected_examples.data.cpu().numpy())
            # np.save(res_path + '/' + 'selected_examples_labels_train.npy', selected_examples_labels.data.cpu().numpy())


    del train_loader, test_loader, trainset, model, optimizer, scheduler
    torch.cuda.empty_cache()
    wandb.finish()



if __name__ == "__main__":
    args = parse_args()
    main(args)
    

