import os
import numpy as np
import pandas as pd
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
from statistics import mean


class training(object):
    def __init__(self, config, data_loader):

        # Data loader
        self.data_loader = data_loader
        self.device = 'cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu'

        # Models
        self.net = None
        self.optimizer = None
        pos_weight = torch.FloatTensor([config.pos_weight])
        if self.device == 'cuda':
            pos_weight = torch.cuda.FloatTensor([config.pos_weight])
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        self.model_type = config.model_type

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.num_epochs_decay = config.num_epochs_decay
        self.classification_threshold = config.classification_threshold

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode


        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='ResNet':
            model = torchvision.models.resnet50(pretrained=True)

            model.fc = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=2048,
                    out_features=1
                )#,
                #torch.nn.Sigmoid()
            )
            self.net = model
        elif self.model_type =='DenseNet':
            model = torchvision.models.densenet121(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, 1)
            self.net = model
        elif self.model_type =='VGG':
            model = torchvision.models.vgg11_bn(pretrained=True)

            num_ftrs = model.classifier[6].in_features
            features = list(model.classifier.children())[:-1]
            features.extend([torch.nn.Linear(num_ftrs, 1)])
            #features.extend([torch.nn.Sigmoid()])
            model.classifier = torch.nn.Sequential(*features)
            self.net = model
            

        self.optimizer = optim.Adam(list(self.net.parameters()),self.lr, [self.beta1, self.beta2])
        self.net.to(self.device)


    def train(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#
        loss_data_perstep = pd.DataFrame()
        loss_data_perepoch = pd.DataFrame()
        lr = self.lr
            
        for epoch in range(self.num_epochs):

            for phase in ["train", "valid"]:
                if self.mode == "train":
                    self.net.train()
                else:
                    self.net.eval()
                
                samples = 0
                loss_sum = 0
                correct_sum = 0
                all_tn = all_fp = all_fn = all_tp = 0
                all_f1 = []
                for j, batch in enumerate(self.data_loader[phase]):
                    X = batch["image"]
                    labels = batch["label"]
                    if self.device == 'cuda':
                        X = X.cuda()
                        labels = labels.cuda()

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        y = self.net(X)
                        loss = self.criterion(
                            y, 
                            labels.view(-1, 1).float()
                        )

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                        loss_sum += loss.item() * X.shape[0] # We need to multiple by batch size as loss is the mean loss of the samples in the batch
                        samples += X.shape[0]
                        tn, fp, fn, tp = confusion_matrix(labels.view(-1, 1).float().cpu(), (y >= self.classification_threshold).float().cpu()).ravel()
                        this_f1 = 2.0*tp/(2.0*tp + fp + fn + 1e-6) 
                        all_f1.append(this_f1)
                        all_tn += tn
                        all_fp += fp
                        all_fn += fn
                        all_tp += tp
                        num_corrects = torch.sum((y >= 0.5).float() == labels.view(-1, 1).float())
                        correct_sum += num_corrects
                        
                        # Print batch statistics every 50 batches
                        if j % 1 == 0 and phase == "train":
                            print("Phase {} - Epoch [{}]: Step[{}] - loss: {}, acc: {}, f1: {}, tn: {}, fp: {}, fn: {}, tp: {}".format(
                                phase,
                                epoch + 1, 
                                j + 1, 
                                round(float(loss_sum) / float(samples), 3), 
                                round(float(correct_sum) / float(samples), 3),
                                round(float(mean(all_f1)), 3),
                                round(float(all_tn) / float(samples), 3),
                                round(float(all_fp) / float(samples), 3),
                                round(float(all_fn) / float(samples), 3),
                                round(float(all_tp) / float(samples), 3)
                            ))
                            loss_data_perstep = loss_data_perstep.append({'phase': phase,
                                                            'epoch': epoch+1,
                                                            'step': j+1,
                                                            'loss': round(float(loss_sum) / float(samples), 3),
                                                            'acc': round(float(correct_sum) / float(samples), 3),
                                                            'f1': round(float(mean(all_f1)), 3),
                                                            'tn': round(float(all_tn) / float(samples), 3),
                                                            'fp': round(float(all_fp) / float(samples), 3),
                                                            'fn': round(float(all_fn) / float(samples), 3),
                                                            'tp': round(float(all_tp) / float(samples), 3)}, ignore_index=True)
                        
                # Print epoch statistics
                epoch_acc = float(correct_sum) / float(samples)
                epoch_loss = float(loss_sum) / float(samples)
                epoch_f1 = float(mean(all_f1))
                epoch_tn = float(all_tn) / float(samples)
                epoch_fp = float(all_fp) / float(samples)
                epoch_fn = float(all_fn) / float(samples)
                epoch_tp = float(all_tp) / float(samples)
                print("epoch: {} - {} loss: {}, {} acc: {}, f1: {}, tn: {}, fp: {}, fn: {}, tp: {}".format(epoch + 1, phase, round(epoch_loss, 3),\
                     phase, round(epoch_acc, 3), round(epoch_f1, 3), round(epoch_tn, 3) \
                         , round(epoch_fp, 3), round(epoch_fn, 3), round(epoch_tp, 3)))
                loss_data_perepoch = loss_data_perepoch.append({'phase': phase,
                                'epoch': epoch+1,
                                'loss': round(float(loss_sum) / float(samples), 3),
                                'acc': round(float(correct_sum) / float(samples), 3),
                                'f1': round(float(all_f1) / float(samples), 3),
                                'tn': round(float(all_tn) / float(samples), 3),
                                'fp': round(float(all_fp) / float(samples), 3),
                                'fn': round(float(all_fn) / float(samples), 3),
                                'tp': round(float(all_tp) / float(samples), 3)}, ignore_index=True)
                # Deep copy the model
                if phase == "valid":# and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    if epoch_acc > best_acc:
                        best_model_wts = copy.deepcopy(self.net.state_dict())
                        torch.save(best_model_wts, os.path.join(self.model_path,"BEST_{}_{}.pth".format(self.model_type, epoch)))

                torch.save(copy.deepcopy(self.net.state_dict()), os.path.join(self.model_path,"{}_{}.pth".format(self.model_type, epoch)))
                loss_data_perstep.to_csv(os.path.join(self.result_path,"loss_data_perstep.csv"))
                loss_data_perepoch.to_csv(os.path.join(self.result_path,"loss_data_perepoch.csv"))


                # Decay learning rate
                if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decay learning rate to lr: {}.'.format(lr))

    def test(self):
        print("To complete testing and calculation of metrics")