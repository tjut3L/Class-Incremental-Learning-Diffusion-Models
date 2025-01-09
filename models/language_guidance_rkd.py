import logging
import numpy as np
import os
import math
import time
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy
from latant_diffusion_model.scripts.txt2img import Text2Img

from utils.classifiers import ProjectionMLP

os.environ["TOKENIZERS_PARALLELISM"] = "false"
EPSILON = 1e-8

class Language_Guidance(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        # self.outdir = self.args['outdir'] + "/ddim/" + self.args["dataset"] + "-photo"
        self.outdir = self.args['outdir'] + "/ddim/" + self.args["dataset"] + "-photo-type"
        # self.outdir = self.args['outdir'] + "/plms/" + self.args["dataset"] + "-photo-attribute"

        self.FT = 1
        self.lambda_ce = 1
        self.lambda_hkd = 3
        self.lambda_rkd = 1
        self.temp = 2
        self.beta_base = 1

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # Loader
        self._learned_task_labels = data_manager.get_learned_task_labels(self._cur_task)
        self._current_task_labels = data_manager.get_current_task_labels(self._cur_task)
        if self._learned_task_labels:
            self._total_task_labels = self._learned_task_labels + self._current_task_labels
        else:
            self._total_task_labels = self._current_task_labels
        # self._current_task_coarse_labels = data_manager.get_current_task_coarse_labels(self._cur_task)
        # self._img_size = data_manager.get_img_size()
        if 'cifar' in self.args["dataset"]:
            self._img_size = (32,32)
        elif 'imagenet100' == self.args["dataset"]:
            self._img_size = (256,256)

        generate_dataset = self._get_generate_dataset(data_manager.use_path)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=generate_dataset,
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"]
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"]
        )

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args['init_lr'],
                weight_decay=self.args['init_weight_decay'],
            )
            scheduler = CosineAnnealingLR(optimizer=optimizer,T_max=self.args["init_epoch"]+1)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            # beta = math.sqrt(self._known_classes / self._cur_classes)
            # self.lambda_hkd = self.beta_base * beta
            info = "lambda_ce {:.4f}, lambda_hkd {:.4f}, lambda_rkd {:.4f}".format(
                    self.lambda_ce,
                    self.lambda_hkd,
                    self.lambda_rkd,
                )
            logging.info(info)
            params = [
                {
                    "params": self._network.convnet.parameters(),
                    "lr": self.args['lr'],
                    "momentum": 0.9,
                    "weight_decay": self.args['weight_decay'],
                },
                {
                    "params": self._network.fc.parameters(),
                    "lr": self.args['lr']/10,
                    "momentum": 0.9,
                    "weight_decay": self.args['weight_decay'],
                }
            ]
            optimizer = optim.SGD(params)
            scheduler = CosineAnnealingLR(optimizer=optimizer,T_max=self.args["epochs"]+1)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        # model_state_dict= torch.load('cifar100/language_guidance_rkd2_50_5_0.pkl')['model_state_dict']
        # self._network.load_state_dict(model_state_dict,strict=True)
        prog_bar = tqdm(range(self.args['init_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

            logging.info(info)
    
    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['epochs']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                output = self._network(inputs)
                logits, features = output["logits"], output["features"]
                old_output = self._old_network(inputs)
                old_out, old_features = old_output["logits"], old_output["features"]


                new_idxes = np.where(np.logical_and(targets >= self._known_classes, targets < self._total_classes))[0]
                old_idxes = np.where(np.logical_and(targets >= 0, targets < self._known_classes))[0]

                loss_clf = self.lambda_ce * F.cross_entropy(logits[new_idxes][:,self._known_classes:], targets[new_idxes].to(self._device)-self._known_classes)

                # loss_hkd = self.lambda_hkd * self._HKD_loss(logits[old_idxes][:,:self._known_classes], old_out[old_idxes], T=2)
                loss_hkd = self.lambda_hkd * self._HKD_loss(logits[old_idxes][:,:self._known_classes], old_out[old_idxes])
                # loss_hkd = self.lambda_hkd * self._HKD_loss(logits[:,:self._known_classes], old_out)

                loss_rkd = self.lambda_rkd * self._RKD_loss(features[new_idxes], old_features[new_idxes])

                targets = targets.to(self._device)

                loss_ft = self.FT * F.cross_entropy(self._network.fc(features.detach())["logits"]/self.temp, targets)
                # loss_ce = self.FT * F.cross_entropy(logits/self.temp, targets)

                loss = loss_clf + loss_hkd + loss_rkd + loss_ft
                # loss = loss_clf + loss_hkd  + loss_ft
                # loss =  loss_hkd  + loss_ce
                # loss = loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['epochs'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['epochs'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

    # def _HKD_loss(self, pred, soft, T):
    #     pred = torch.log_softmax(pred / T, dim=1)
    #     soft = torch.softmax(soft / T, dim=1)
    #     return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
    
    def _HKD_loss(self, pred, soft):
        g = torch.sigmoid(pred)
        q_i = torch.sigmoid(soft)
        return sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in
                                        range(self._known_classes))
    
    def _RKD_loss(self, student, teacher):
        td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss

    def _get_generate_dataset(self, use_path):
        if self._learned_task_labels is None :
            return None
        else:
            sample_data = []
            sample_target = []
            for i in range(len(self._learned_task_labels)):
                class_data = []
                sample_path = os.path.join(self.outdir, self._learned_task_labels[i])
                for filename in os.listdir(sample_path):
                    img_path = os.path.join(sample_path, filename)
                    if not use_path:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self._img_size)
                        class_data.append(img)
                    else:
                        class_data.append(img_path)
                sample_data.extend(class_data[:self.args['n_sum']])
                sample_target.extend([i]* self.args['n_sum'])
            sample_data = np.array(sample_data)
            sample_target = np.array(sample_target)
            return (sample_data, sample_target)
    
    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        # self.save_checkpoint("{}_{}_{}".format(self.args["dataset"],self.args["init_cls"],self.args["increment"]))
        # start = time.time()
        # text2img=Text2Img(self._current_task_labels, 
        #                   self._current_task_coarse_labels, 
        #                   self.outdir, 
        #                   self.args['steps'], 
        #                   self.args['use_plms'], 
        #                   self.args['ddim_eta'],
        #                   self.args['n_sum'],
        #                   self.args['n_samples'],
        #                   self.args['scale'],
        #                   self.args['H'],
        #                   self.args['W'],
        #         )
        # text2img.generate_img()
        # end = time.time()
        # info = "complete generate old class images time={:5.1f}s".format(end-start)
        # logging.info(info)   
