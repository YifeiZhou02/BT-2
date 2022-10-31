#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Tuple, Callable

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.logging_utils import AverageMeter
from utils.eval_utils import accuracy

Temperature = 1


class RegularizedTrainer():
    """Class to train and evaluate regularized new model 
    with a given old model."""

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              model: nn.Module,
              criterion: Callable,
              old_feature_dict,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              accelerator) -> Tuple[float, float, float]:
        """Run one epoch of training.

        :param train_loader: Data loader to train the model.
        :param model: Model to be trained.
        :param criterion: Loss criterion module.
        :param optimizer: A torch optimizer object.
        :param device: Device the model is on.
        :return: average of top-1, top-5, and loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")
        deviation_losses = AverageMeter("Deviation Loss", ":.3f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        model = model.train()

        for i, ((paths, images), target) in tqdm.tqdm(
                enumerate(train_loader), ascii=True, total=len(train_loader)
        ):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output, feature = model(images)
            feature = feature.view(feature.size(0), -1)
            valid = (target.reshape(1, -1) !=
                     target.reshape(-1, 1)).fill_diagonal_(1)

            old_feature = [old_feature_dict[path].view(
                1, -1) for path in paths]
            old_feature = torch.cat(old_feature, dim=0).to(device)

            old_sim = feature @ old_feature.transpose(0, 1)
            old_sim = torch.exp(old_sim/Temperature)*valid
            new_sim = feature @ feature.transpose(0, 1)
            new_sim = torch.exp(new_sim/Temperature) * \
                (target.reshape(1, -1) != target.reshape(-1, 1))
            sim_loss = -torch.mean(torch.log(torch.diagonal(old_sim /
                                   (torch.sum(old_sim, dim=1) + torch.sum(new_sim, dim=1)))))

            loss = 10*sim_loss + criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        return losses.avg

    def validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 model: nn.Module,
                 criterion: Callable,
                 device: torch.device) -> Tuple[float, float, float]:
        """Run validation.

        :param val_loader: Data loader to evaluate the model.
        :param model: Model to be evaluated.
        :param criterion: Loss criterion module.
        :param device: Device the model is on.
        :return: average of top-1, top-5, and loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")
        deviation_losses = AverageMeter("Deviation Loss", ":.3f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        model = model.eval()

        with torch.no_grad():
            for i, ((_, images), target) in tqdm.tqdm(
                    enumerate(val_loader), ascii=True, total=len(val_loader)
            ):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output, feature = model(images)

                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

        return losses.avg, top1.avg
