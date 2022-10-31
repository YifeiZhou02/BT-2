#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Tuple, Callable

import tqdm
import torch
import torch.nn as nn

from utils.logging_utils import AverageMeter
from utils.eval_utils import accuracy


class BackboneTrainer():
    """Class to train and evaluate backbones."""

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              model: nn.Module,
              criterion: Callable,
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
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        model.train()

        # embedding_dim = model.fc.weight.size(0)

        # assert embedding_dim == 128

        for i, ((paths, images), target) in tqdm.tqdm(
                enumerate(train_loader), ascii=True, total=len(train_loader)
        ):
            # try:
            #     images = model.feature_extractor(images)
            # except AttributeError:
            #     pass
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output, features = model(images)
            # stable = 1e-4*torch.eye(features.size(1)).to(device)
            features = features.view(features.size(0), -1)
            # aux_features = features.view(features.size(0), features.size(1),1,1)
            # aux_features[:, features.size(1)//2:] = 0
            loss = criterion(output, target)

            # feature_covariance = (features.transpose(0,1) @ features)/\
            #     images.size(0)
            # eigenvalues, eigenvectors = torch.linalg.eigh(feature_covariance)
            # # half_eigenvectors = eigenvectors[:, :eigenvectors.size(1)//2]
            # # aux_features = features @ half_eigenvectors @ half_eigenvectors.transpose(0,1)
            # # aux_features = aux_features.reshape(features.size(0), features.size(1), 1, 1)
            # # pseudo_output = model.module.fc(aux_features).view(output.size())
            # # pseudo_loss = criterion(pseudo_output, target)
            # # half_eigenvalues = eigenvalues[-eigenvalues.size(0)//2:]

            # loss_of_eigens = - torch.sum(eigenvalues * eigenvalues)

            # lasso_loss = torch.mean(torch.abs(features))

            loss = loss  # + loss_of_eigens

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()

        return top1.avg, top5.avg, losses.avg

    def validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 model: nn.Module,
                 criterion: Callable,
                 device: torch.device,
                 accelerator) -> Tuple[float, float, float]:
        """Run validation.

        :param val_loader: Data loader to evaluate the model.
        :param model: Model to be evaluated.
        :param criterion: Loss criterion module.
        :param device: Device the model is on.
        :return: average of top-1, top-5, and loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        model.eval()

        with torch.no_grad():
            for i, ((paths, images), target) in tqdm.tqdm(
                    enumerate(val_loader), ascii=True, total=len(val_loader)
            ):
                # try:
                #     images = model.feature_extractor(images)
                # except AttributeError:
                #     pass
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output, _ = model(images)

                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

        return top1.avg, top5.avg, losses.avg
