import torch
import  torch.nn.functional as F

import wandb


def compute_features(args,net,trainloader,testtransform,device, epoch, ret_pred=False):
    net.eval()

    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testtransform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers_sel)
        for batch_idx, (inputs, _,_) in enumerate(temploader):
            
            batchSize = inputs.size(0)
            inputs = inputs.cuda()

            output,features = net(inputs)
            output = F.softmax(output, -1)
            if batch_idx == 0:
                trainFeatures = features.data
            else:
                trainFeatures = torch.cat((trainFeatures, features.data), 0)

            if batch_idx == 0:
                model_preds = output.cpu()
            else:
                model_preds = torch.cat((model_preds, output.cpu()), 0)

                    
    trainloader.dataset.transform = transform_bak
    clean_labels = torch.LongTensor(trainloader.dataset.clean_labels)
    train_acc = (clean_labels == torch.max(model_preds,-1)[1]).sum()/clean_labels.shape[0]

    print('Train Accuracy:', train_acc)

    wandb.log({
            'Train Accuracy': train_acc
        }, epoch)

    if ret_pred:
        return trainFeatures.cpu(), model_preds
    return trainFeatures.cpu()