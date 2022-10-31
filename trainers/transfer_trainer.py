#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from math import sqrt
from typing import Dict, Tuple, Callable

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logging_utils import AverageMeter
from utils.eval_utils import accuracy

cosine_criterion = nn.CosineSimilarity(dim=1)
entropy_criterion = nn.CrossEntropyLoss()
kl_criterion = nn.KLDivLoss()
mse_criterion = nn.MSELoss()
Temperature = .2
sf = nn.Softmax(dim=1)


class TransferTrainer():
    """Class to train and evaluate regularized new model 
    with a given old model."""

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              model: nn.Module,
              old_feature_dict: Dict,
              new_feature_dict: Dict,
              criterion: Callable,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              accelerator,
              pseudo_classifier=None) -> Tuple[float, float, float]:
        """Run one epoch of training.

        :param train_loader: Data loader to train the model.
        :param model: Model to be trained.
        :param criterion: Loss criterion module.
        :param optimizer: A torch optimizer object.
        :param device: Device the model is on.
        :return: average of top-1, top-5, and loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")

        model = model.train().to(device)
        # new_model = new_model.eval().to(device)

        for i, ((paths, images), target) in tqdm.tqdm(
                enumerate(train_loader), ascii=True, total=len(train_loader)
        ):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # with torch.no_grad():
            #     old_result = old_model(images)
            #     if len(old_result) == 2:
            #         phi_old = old_result[1]
            #     else:
            #         phi_old = old_model(images)
            #     phi_old = phi_old.view(phi_old.size(0), -1)

            # try:
            #     images = model.feature_extractor(images.cpu()).to(device, non_blocking=True)
            # except AttributeError:
            #     pass

            # valid_flags = (target.reshape(1,-1) != target.reshape(-1,1))
            # valid_flags = valid_flags.fill_diagonal_(1)

            # assert valid_flags.size() == (target.size(0), target.size(0))
            outputs = model(images)
            old_feature = outputs[1]
            feature = outputs[2]
            output = outputs[0]
            feature = feature.view(feature.size(0), -1)
            old_feature = old_feature.view(feature.size(0), -1)
            old_feature = F.normalize(old_feature, dim=1)
            feature = F.normalize(feature, dim=1)

            # if images[0].cpu().numpy() in old_featxure_dict:
            #     print('using dictionary')
            #     old_feature = [torch.Tensor(old_feature_dict[image]) for image in images.cpu().numpy()]
            #     old_feature = torch.cat(old_feature, dim = 0).to(device)
            # else:
            #     with torch.no_grad():
            #         old_feature = old_model(images)
            #     old_feature = old_feature.view(old_feature.size(0), -1)
            #     for i, image in enumerate(images.cpu().numpy()):
            #         old_feature_dict[image] = old_feature[i].reshape(1,-1).cpu().numpy()
            # with torch.no_grad():
            #     _, phi_p = new_model(images)
            #     phi_p = phi_p.view(phi_p.size(0), -1)

            phi_old = []
            phi_p = []
            for path in paths:
                phi_old.append(old_feature_dict[path].view(1, -1))
                phi_p.append(new_feature_dict[path].view(1, -1))
            phi_old = torch.cat(phi_old, dim=0).to(device)
            phi_p = torch.cat(phi_p, dim=0).to(device)

            cosine_loss = 10 * \
                (1 - torch.mean(torch.sum(phi_old * old_feature, dim=1)))
            cosine_loss += 1 * \
                (1 - torch.mean(torch.sum(phi_p * feature, dim=1)))

            loss = entropy_criterion(output, target)
            if pseudo_classifier is not None:
                pseudo_output = old_feature.view(old_feature.size(
                    0), -1) @ pseudo_classifier.transpose(0, 1)
                loss += 10*entropy_criterion(pseudo_output, target)
            loss += cosine_loss

            # loss = kl_criterion(old_feature_sim, new_feature_sim)
            # loss = entropy_criterion(pseudo_output, target)

            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        return losses.avg

    def validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 model: nn.Module,
                 old_feature_dict: Dict,
                 new_feature_dict: Dict,
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
        top1 = AverageMeter("Acc@1", ":6.2f")

        model = model.eval()
        # new_model = new_model.eval()

        with torch.no_grad():
            for i, ((paths, images), target) in tqdm.tqdm(
                    enumerate(val_loader), ascii=True, total=len(val_loader)
            ):
                # try:
                #     images = model.feature_extractor(images.cpu()).to(device, non_blocking=True)
                # except AttributeError:
                #     pass
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                model = model.to(device)

                outputs = model(images)
                old_feature = outputs[1]
                feature = outputs[2]
                output = outputs[0]
                feature = feature.view(feature.size(0), -1)
                old_feature = old_feature.view(feature.size(0), -1)
                feature = F.normalize(feature, dim=1)
                old_feature = F.normalize(old_feature, dim=1)
            # pseudo_output = feature.view(feature.size(0) ,-1) @ pseudo_classifier.transpose(0,1)

            # if images[0].cpu().numpy() in old_featxure_dict:
            #     print('using dictionary')
            #     old_feature = [torch.Tensor(old_feature_dict[image]) for image in images.cpu().numpy()]
            #     old_feature = torch.cat(old_feature, dim = 0).to(device)
            # else:
            #     with torch.no_grad():
            #         old_feature = old_model(images)
            #     old_feature = old_feature.view(old_feature.size(0), -1)
            #     for i, image in enumerate(images.cpu().numpy()):
            #         old_feature_dict[image] = old_feature[i].reshape(1,-1).cpu().numpy()
                phi_old = []
                phi_p = []
                for path in paths:
                    phi_old.append(old_feature_dict[path].view(1, -1))
                    phi_p.append(new_feature_dict[path].view(1, -1))
                phi_old = torch.cat(phi_old, dim=0).to(device)
                phi_p = torch.cat(phi_p, dim=0).to(device)

            # # old_output = model.module.fc(old_feature).view(output.size())
            # old_feature = old_feature.view(old_feature.size(0), -1)
            # feature = feature.view(feature.size(0), -1)
            # # learning_loss = entropy_criterion(output, target) + entropy_criterion(old_output, target)
            # # mse_loss = mse_criterion(old_feature, feature)
            # # logits = sf(output/Temperature)
            # # old_logits = sf(old_output/Temperature)
            # # distill_loss = kl_criterion(logits, old_logits)

            # # loss = cosine_loss + learning_loss + mse_loss + distill_loss

            # # old_feature_sim = sf(old_feature @ old_feature.transpose(0,1))
            # new_feature_sim = torch.exp((feature @ old_feature.transpose(0,1))/Temperature)

            # denominator = torch.sum(valid_flags * new_feature_sim, dim = 1)

            # new_feature_sim = new_feature_sim/denominator.reshape(feature.size(0), 1)
                loss = 1 - torch.mean(torch.sum(phi_p * feature, dim=1))
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), images.size(0))
                # cosine_loss += 1 - torch.mean(torch.sum(phi_p * feature, dim = 1))
            # loss = - torch.trace(torch.log(new_feature_sim))/feature.size(0)

                # loss = entropy_criterion(output, target)

                # loss = - torch.trace(torch.log(new_feature_sim))/feature.size(0)
                # loss = 1 - torch.mean(criterion(feature.view(old_feature.size()), old_feature))

                losses.update(loss.item(), images.size(0))

        return losses.avg, top1.avg


def build_feature_dict(loader: torch.utils.data.DataLoader,
                       model: nn.Module,
                       device: torch.device,
                       feature_dict) -> Tuple[float, float, float]:
    """
    return a dictionary of saved features
    """
    model = model.eval().to(device)
    # new_model = new_model.eval().to(device)

    for i, ((paths, images), target) in tqdm.tqdm(
            enumerate(loader), ascii=True, total=len(loader)
    ):
        images = images.to(device, non_blocking=True)
        model = model

        try:
            images = model.feature_extractor(
                images.cpu()).to(device, non_blocking=True)
        except AttributeError:
            pass

        with torch.no_grad():
            features = model(images)[-1].cpu()
        features = features.reshape(features.size(0), -1)
        features = F.normalize(features, dim=1)
        for path, feature in zip(paths, features):
            feature_dict[path] = feature

    return feature_dict


def build_text_feature_dict(loader: torch.utils.data.DataLoader,
                            model: nn.Module,
                            device: torch.device,
                            feature_dict,
                            path2caption,
                            tokenizer) -> Tuple[float, float, float]:
    """
    return a dictionary of saved features
    """
    model = model.eval().to(device)
    # new_model = new_model.eval().to(device)

    for i, ((paths, images), target) in tqdm.tqdm(
            enumerate(loader), ascii=True, total=len(loader)
    ):
        with torch.no_grad():
            features = get_text_features(paths, path2caption, tokenizer, device,
                                         model)
            features = features.reshape(features.size(0), -1)
            for path, feature in zip(paths, features):
                feature_dict[path] = feature

    return feature_dict


def get_text_features(paths, path2caption, tokenizer, device, vit_model):
    captions = [path2caption[p] for p in paths]
    inputs = tokenizer(captions, padding=True, return_tensors="pt").to(device)
    features = F.normalize(vit_model.get_text_features(**inputs))
    return features
