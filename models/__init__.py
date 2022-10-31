#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from .resnet import ResNet18, ResNet50, ResNet101, WideResNet50_2, \
    WideResNet101_2, Backbone, BackboneBase,\
        ResNet_ortho, ResNet50_ortho
from .transformers import VIT_google, VIT, VIT_google_shallow, VIT_shallow,\
    VIT_ortho_shallow, VIT_google_ortho_shallow, VIT_pretrained, VIT_CLIP_pretrained
