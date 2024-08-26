import json
import numpy as np
import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset

import wandb

def get_dataset(args, transform_train, transform_test):
    if args.slice == 1:
        train_split, test_split = ['fold1','fold4','fold5'],['fold3']
    elif args.slice == 2:
        train_split, test_split = ['fold1','fold2','fold5'],['fold4']
    elif args.slice == 3:
        train_split, test_split = ['fold1','fold2','fold3'],['fold5']
    trainset = Crowdsource(args, splits=train_split, transform=transform_train)
    trainset.partial_noise()
    testset = Crowdsource(args, splits=test_split, transform=transform_test)
    return trainset, testset

class Crowdsource(Dataset):

    def __init__(self, args, splits=['fold1'], transform=None):
        self.root = os.path.expanduser(args.train_root)
        # self.root = './Treeversity#6'
        self.transform = transform
        self.args = args
        self.num_classes = self.args.num_classes
        self.noisy_labels = []
        self.train = len(splits)!=1

        annotation_file = self.root + '/annotations.json'
        with open(annotation_file, 'r') as outfile:
            annotation_jsons = json.load(outfile)
            
        img_names = []
        labels = []
        for entry in annotation_jsons[0]["annotations"]:
        # add only valid annotations to table
            if entry["class_label"] is not None:
                # if entry["image_path"] not in img_names:
                img_names.append(entry["image_path"])
                # if entry["class_label"] not in img_names:
                labels.append(entry["class_label"])
                
        img_names = list(np.unique(np.array(img_names)))
        labels = list(np.unique(np.array(labels)))

        # fast access maps
        map_names = dict(zip(img_names, list(np.arange(0, len(img_names)))))
        map_labels = dict(zip(labels, list(np.arange(0, len(labels)))))
        
        self.folds = {}
        
        for name in img_names:
            fold = name.split('/')[1]
            if fold in self.folds:
                self.folds[fold].append(map_names[name])
            else:
                self.folds[fold] = [map_names[name]]
        
        _data = np.zeros((len(img_names), len(labels)))

        for entry in annotation_jsons[0]["annotations"]:
            # add only valid annotations to table
            if entry["class_label"] is not None:
                _data[map_names[entry["image_path"]], map_labels[entry["class_label"]]] += 1

        data = np.zeros((len(img_names), len(labels)))
        rng = np.random.default_rng(self.args.seed_dataset)
        for i in range(data.shape[0]):
            annots = rng.choice(args.num_classes,p=_data[i]/_data[i].sum(),size=args.lpi)
            for a in annots:
                data[i,a] += 1
            assert data[i].sum() == args.lpi
                
                
        partialY = np.zeros((len(img_names), len(labels)))
        partialY[data.nonzero()] = 1
        
        weights = data/data.sum(1,keepdims=True)
        clean_labels = np.argmax(_data,axis=1)
        
        req_ids = []
        for split in splits:
            req_ids.extend(self.folds[split])
        

        self.soft_labels = partialY[req_ids]
        self.clean_labels = clean_labels[req_ids]
        self.data = [img_names[i] for i in req_ids]
        self.weights = weights[req_ids]
        self.targets = np.copy(self.clean_labels)
        majority_label = np.argmax(self.weights,axis=1)
        clean_majority = sum(majority_label == self.clean_labels)
                
        print('Average candidate num: ', self.soft_labels.sum(1).mean())
        print('clean_num', 
              sum(self.soft_labels[range(len(self.clean_labels)),self.clean_labels] == 1),'/',len(self.clean_labels))
        print('clean majority', clean_majority)
        wandb.log({
            'total_labels': len(self.clean_labels),
            'clean_labels': sum(self.soft_labels[range(len(self.clean_labels)),self.clean_labels] == 1),
            'clean_majority_labels': clean_majority
        })

    def partial_noise(self):
        self.targets = np.zeros((len(self.clean_labels),))-1

    def __getitem__(self, index):

        img, labels = self.data[index], self.targets[index]
        img = Image.open(img)
        img = self.transform(img)
        if self.train:
            return img, labels, index
        else:
            return img, labels
        
    def __len__(self):
        return len(self.targets)
    