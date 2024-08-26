import os
import pickle
import torchvision as tv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

def get_dataset(args, transform_train, transform_test):

    if args.dataset == 'CIFAR-100':
        trainset = CIFAR100Partial(args, transform=transform_train, target_transform=transform_test, download=args.download)
        testset = tv.datasets.CIFAR100(root=args.train_root, train=False, download=args.download, transform=transform_test)
        if args.noise_type == 'partial':
            trainset.partial_noise(partial_rate=args.partial_ratio, noisy_rate=args.noise_ratio, heirarchical=args.heirarchical)
        

    elif args.dataset == 'CIFAR-10':
        trainset = CIFAR10Partial(args, transform=transform_train, target_transform=transform_test, download=args.download)
        testset = tv.datasets.CIFAR10(root=args.train_root, train=False, download=args.download, transform=transform_test)
        if args.noise_type == 'partial':
           trainset.partial_noise(partial_rate=args.partial_ratio, noisy_rate=args.noise_ratio)

    elif args.dataset == 'CUB-200':
        trainset = CUB200Partial(args, train=True, transform=transform_train)
        testset = CUB200Partial(args, train=False, transform=transform_test)
        if args.noise_type == 'partial':
            trainset.partial_noise(partial_rate=args.partial_ratio, noisy_rate=args.noise_ratio)
       
    return trainset, testset


class CUB200Partial(Dataset):

    def __init__(self, args, train=True, transform=None):
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.train = train
        self.args = args
        self.num_classes = self.args.num_classes

        if self.train:
            self.data, self.targets = pickle.load(open(
                os.path.join(self.root, 'processed/train.pkl'), 'rb'))
            assert (len(self.data) == 5994 and len(self.targets) == 5994)
        else:
            self.data, self.targets = pickle.load(open(
                os.path.join(self.root, 'processed/test.pkl'), 'rb'))
            assert (len(self.data) == 5794 and len(self.targets) == 5794)


        self.targets = np.array(self.targets)

    def partial_noise(self, partial_rate, noisy_rate):
        np.random.seed(self.args.seed_dataset)
        self.clean_labels = np.copy(self.targets)
        clean_labels = np.copy(self.targets)


        partialY = generate_uniform_cv_candidate_labels(self.clean_labels, partial_rate, noisy_rate=noisy_rate)

        self.soft_labels = np.asarray(partialY)
        
        temp = torch.zeros(partialY.shape)
        temp[torch.arange(partialY.shape[0]), clean_labels] = 1
        if torch.sum(partialY * temp) == partialY.shape[0]:
            print('partialY correctly loaded')
        else:
            print('inconsistent permutation')
        print('Average candidate num: ', partialY.sum(1).mean())

        self.targets = np.zeros((len(self.targets),))-1
        print('clean_num', 
              sum(self.soft_labels[range(len(self.clean_labels)),self.clean_labels] == 1))



    def __getitem__(self, index):

        img, labels = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        if self.train:
            return img, labels, index
        else:
            return img, labels
        
    def __len__(self):
        return len(self.targets)
    

class CIFAR10Partial(tv.datasets.CIFAR10):

    def __init__(self, args, train=True, transform=None, target_transform=None, sample_indexes=None, download=False):
        super(CIFAR10Partial, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        self.args = args
        if sample_indexes is not None:
            self.data = self.data[sample_indexes]
            self.targets = list(np.asarray(self.targets)[sample_indexes])
        
        self.num_classes = self.args.num_classes

    def partial_noise(self, partial_rate, noisy_rate):
        np.random.seed(self.args.seed_dataset)
        self.clean_labels = np.copy(self.targets)
        clean_labels = np.copy(self.targets)


        partialY = generate_uniform_cv_candidate_labels(self.clean_labels, partial_rate, noisy_rate=noisy_rate)

        self.soft_labels = np.asarray(partialY)
        
        temp = torch.zeros(partialY.shape)
        temp[torch.arange(partialY.shape[0]), clean_labels] = 1
        if torch.sum(partialY * temp) == partialY.shape[0]:
            print('partialY correctly loaded')
        else:
            print('inconsistent permutation')
        print('Average candidate num: ', partialY.sum(1).mean())

        self.targets = np.zeros((len(self.targets),))-1
        print('clean_num', 
              sum(self.soft_labels[range(len(self.clean_labels)),self.clean_labels] == 1))
        #print('clean_num',sum(self.targets==self.clean_labels))


    def __getitem__(self, index):
        if self.train:
            img, labels = self.data[index], self.targets[index]
            
            img = Image.fromarray(img)

            img1 = self.transform(img)
            
            return img1, labels, index

        else:
            img, labels = self.data[index], self.targets[index]
            # doing this so that it is consistent with all other datasets.
            img = Image.fromarray(img)

            img = self.transform(img)

            return img, labels




class CIFAR100Partial(tv.datasets.CIFAR100):

    def __init__(self, args, train=True, transform=None, target_transform=None, sample_indexes=None, download=False):
        super(CIFAR100Partial, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        self.args = args
        if sample_indexes is not None:
            self.data = self.data[sample_indexes]
            self.targets = list(np.asarray(self.targets)[sample_indexes])
        
        self.num_classes = self.args.num_classes
        self.clean_labels = np.copy(self.targets)

    def partial_noise(self, partial_rate, noisy_rate, heirarchical):
        np.random.seed(self.args.seed_dataset)
        self.clean_labels = np.copy(self.targets)
        clean_labels = np.copy(self.targets)

        if heirarchical:
            partialY = generate_hierarchical_cv_candidate_labels('cifar100', self.clean_labels, partial_rate, noisy_rate=noisy_rate)
        else:
            partialY = generate_uniform_cv_candidate_labels(self.clean_labels, partial_rate, noisy_rate=noisy_rate)

        self.soft_labels = np.asarray(partialY)
        
        temp = torch.zeros(partialY.shape)
        temp[torch.arange(partialY.shape[0]), clean_labels] = 1
        if torch.sum(partialY * temp) == partialY.shape[0]:
            print('partialY correctly loaded')
        else:
            print('inconsistent permutation')
        print('Average candidate num: ', partialY.sum(1).mean())

        self.targets = np.zeros((len(self.targets),))-1
        print('clean_num', 
              sum(self.soft_labels[range(len(self.clean_labels)),self.clean_labels] == 1))



    def __getitem__(self, index):
        if self.train:
            img, labels = self.data[index], self.targets[index]
            
            img = Image.fromarray(img)

            img1 = self.transform(img)
            
            return img1, labels, index

        else:
            img, labels = self.data[index], self.targets[index]
            # doing this so that it is consistent with all other datasets.
            img = Image.fromarray(img)

            img = self.transform(img)

            return img, labels



def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res



def generate_hierarchical_cv_candidate_labels(dataname, train_labels, partial_rate=0.1, noisy_rate=0):
    train_labels = torch.tensor(train_labels)
    assert dataname == 'cifar100'

    meta = unpickle('dataset/cifar-100-python/meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]:i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
    fish#aquarium fish, flatfish, ray, shark, trout
    flowers#orchid, poppy, rose, sunflower, tulip
    food containers#bottle, bowl, can, cup, plate
    fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
    household electrical devices#clock, keyboard, lamp, telephone, television
    household furniture#bed, chair, couch, table, wardrobe
    insects#bee, beetle, butterfly, caterpillar, cockroach
    large carnivores#bear, leopard, lion, tiger, wolf
    large man-made outdoor things#bridge, castle, house, road, skyscraper
    large natural outdoor scenes#cloud, forest, mountain, plain, sea
    large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
    medium-sized mammals#fox, porcupine, possum, raccoon, skunk
    non-insect invertebrates#crab, lobster, snail, spider, worm
    people#baby, boy, girl, man, woman
    reptiles#crocodile, dinosaur, lizard, snake, turtle
    small mammals#hamster, mouse, rabbit, shrew, squirrel
    trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
    vehicles 1#bicycle, bus, motorcycle, pickup truck, train
    vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]
            
        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    transition_matrix = np.eye(K) * (1 - noisy_rate)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        random_n_j = random_n[j]
        while partialY[j].sum() == 0:
            random_n_j = np.random.uniform(0, 1, size=(1, K))
            partialY[j] = torch.from_numpy((random_n_j <= transition_matrix[train_labels[j]]) * 1)
    
    print("Finish Generating Heirarchical Candidate Label Sets!\n")
    return partialY



def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1, noisy_rate=0):
    
    train_labels = torch.tensor(train_labels)
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    # partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K) * (1 - noisy_rate)
    # inject label noise if noisy_rate > 0
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        random_n_j = random_n[j]
        while partialY[j].sum() == 0:
            random_n_j = np.random.uniform(0, 1, size=(1, K))
            partialY[j] = torch.from_numpy((random_n_j <= transition_matrix[train_labels[j]]) * 1)

    if noisy_rate == 0:
        partialY[torch.arange(n), train_labels] = 1.0
        # if supervised, reset the true label to be one.
        print('Reset true labels')

    print("Finish Generating Candidate Label Sets!\n")
    return partialY

