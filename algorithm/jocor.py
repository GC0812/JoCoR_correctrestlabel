# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.cnn import MLPNet,CNN
import numpy as np
from common.utils import accuracy

from algorithm.loss import loss_jocor

import copy

class JoCoR:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = 128
        learning_rate = args.lr

        if args.forget_rate is None:
            if args.noise_type == "asymmetric":
                forget_rate = args.noise_rate / 2
            else:
                forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate

        self.noise_or_not = train_dataset.noise_or_not
        self.train_dataset = train_dataset

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)
        ##TODO define correction rate schedule
        self.epoch_loop = 10
        self.rate_schedule=self.rate_schedule[:self.epoch_loop]


        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        print('self.num_iter_per_epoch',self.num_iter_per_epoch)
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset

        if args.model_type == "cnn":
            self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        elif args.model_type == "mlp":
            self.model1 = MLPNet()
            self.model2 = MLPNet()

        self.model1.to(device)
        print(self.model1.parameters)

        self.model2.to(device)
        print(self.model2.parameters)

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=learning_rate)

        self.loss_fn = loss_jocor


        self.adjust_lr = args.adjust_lr

    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')
        num_of_batch = 50 #gc
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        correct1 = 0
        total1 = 0

        # for images, labels, _ in test_loader: #gc
        for i, (images, labels, _) in enumerate(test_loader):
            if i > num_of_batch:
                break
            images = Variable(images).to(self.device)
            logits1 = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

        correct2 = 0
        total2 = 0
        print('model1 done.')

        for j, (images, labels, _) in enumerate(test_loader):
        # for images, labels, _ in test_loader:
            if j > num_of_batch:
                break
            images = Variable(images).to(self.device)
            logits2 = self.model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()
        print('model2 done.')
        acc1 = 100 * float(correct1) / float(total1)
        acc2 = 100 * float(correct2) / float(total2)
        return acc1, acc2

    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []
        num_correction=0
        num_correctlabel = 0
        num_true_correction = 0
        num_false_correction = 0

        _train_labels = [i[0] for i in self.train_dataset.train_labels]
        #self.train_dataset.train_noisy_labels = copy.deepcopy(self.train_dataset.train_noisy_labels_raw)
        for i, (images, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            # print(i)
            if i > self.num_iter_per_epoch:
                break

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            # loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_correction = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch],
            #                                                      ind, self.noise_or_not, self.co_lambda)
            correct_rate = 0.1+epoch//10*0.1
            test = sum(self.noise_or_not)
            loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_correction = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch%self.epoch_loop],
                                                                 correct_rate,ind, self.noise_or_not, self.co_lambda)
            # TODO label correction
            #self.epoch_loop = 1
            if epoch>0 and epoch%self.epoch_loop==0:

                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                outputs2 = F.softmax(logits2, dim=1)
                _, pred2 = torch.max(outputs2.data, 1)
                equalpred = pred1.cpu()[ind_correction]==pred2.cpu()[ind_correction]
                difflabel = pred1.cpu()[ind_correction] != labels[ind_correction]
                update_label_idx = [equalpred[i] and difflabel[i] for i in range(len(equalpred))]
                to_be_corrected=ind_correction[equalpred]

                for idx in to_be_corrected:
                    label = labels[idx]
                    correction = pred1.cpu()[idx].item()
                    self.train_dataset.train_noisy_labels[indexes[idx]] = correction
                    num_correction+=1
                #evaluate correction
                new_noise_or_not = np.transpose(self.train_dataset.train_noisy_labels) == np.transpose(_train_labels)
                tempidx = np.transpose(self.train_dataset.train_noisy_labels)!=self.train_dataset.train_noisy_labels_raw
                true_correction = [new_noise_or_not[i] and tempidx[i] for i in range(len(new_noise_or_not))]
                false_correction = [not new_noise_or_not[i] and tempidx[i] for i in range(len(new_noise_or_not))]
                num_changedlabel = sum(tempidx)
                num_correctlabel = sum(new_noise_or_not)
                num_true_correction = sum(true_correction)
                num_false_correction = sum(false_correction)
                correction_ratio = num_correctlabel / len(_train_labels)

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

            if (i + 1) % self.print_freq == 0:
            # if (i + 1) % 1 == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss: %.4f, Pure Ratio %.4f, num_correctlabel: %d, num_true_correction: %d, num_false_correction: %d'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
                       loss_1.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), num_correctlabel, num_true_correction,num_false_correction))

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)

        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
