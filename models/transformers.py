from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel
from .resnet import BasisChange
import numpy as np
from torch.nn.utils.parametrizations import orthogonal
from transformers import CLIPFeatureExtractor, CLIPVisionModel
from transformers import CLIPTokenizer, CLIPModel


class VIT(nn.Module):
    """
    VIT base model + normalization + linear head for image classification
    """

    def __init__(self,
                 embedding_dim: int = 128,
                 pretrained_name: str = 'openai/clip-vit-base-patch32',
                 num_classes: int = 1000,
                 norm_feature: bool = False) -> None:
        super(VIT, self).__init__()
        self.transformer = ViTModel.from_pretrained(pretrained_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            pretrained_name)
        self.embedding_dim = embedding_dim
        self.embedding_fc = nn.Linear(768, embedding_dim)
        self.output_fc = nn.Linear(embedding_dim, num_classes)
        self.norm_feature = norm_feature

    def forward(self, x):
        tokens = x
        features = self.transformer(tokens).pooler_output
        features = self.embedding_fc(features)
        if self.norm_feature:
            features = F.normalize(features)
        output = self.output_fc(features)
        output = output.view(output.size(0), -1)
        features = features.view(features.size(0), -1)
        return output, features


def VIT_google(num_classes: int,
               embedding_dim: int,
               norm_feature: bool = True,
               **kwargs) -> nn.Module:
    """Get a ViTB16 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :return: ViTB16 Model.
    """
    return VIT(
        num_classes=num_classes,
        pretrained_name='google/vit-base-patch16-224-in21k',
        embedding_dim=embedding_dim,
        norm_feature=norm_feature
    )


class VIT_pretrained(nn.Module):
    """
    Pretrained ViTModel
    """

    def __init__(self,
                 pretrained_name: str = 'openai/clip-vit-base-patch32'
                 ) -> None:
        super(VIT_pretrained, self).__init__()
        self.transformer = ViTModel.from_pretrained(pretrained_name)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
            pretrained_name)

    def forward(self, x):
        x = self.transformer(x).pooler_output
        return x


def VIT_CLIP_pretrained(num_classes: int,
                        embedding_dim: int,
                        norm_feature: bool = True,
                        **kwargs) -> nn.Module:
    """Get a Pretrained CLIP image encoder.
    """
    return VIT_pretrained(
        pretrained_name='openai/clip-vit-base-patch32'
    )


class VIT_ortho_shallow(nn.Module):
    """
    ViT + a series of Basis Transformations
    """

    def __init__(self,
                 embedding_dim: int = 128,
                 pretrained_name: str = 'openai/clip-vit-base-patch32',
                 num_classes: int = 1000,
                 norm_feature: bool = False,
                 to_add_dim: int = 32,
                 C: float = 3) -> None:
        """
        Construct a ViT + Basis Transformations model.
        :param embedding_dim: the dimension of the embedding (same
            for new' representations and old representations).
        :param pretrained_name: the name of the pretrained model.
        num_classes: number of classes for classification.
        norm_feature: whether to normalize features before linear
            classification.
        to_add_dim: the number of additional dimensions in the final
            new representation (so size of the new representation will
            be embedding_dim + to_add_dim)
        """
        super(VIT_ortho_shallow, self).__init__()
        self.transformer = ViTModel.from_pretrained(pretrained_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            pretrained_name)
        self.embedding_dim = embedding_dim
        self.norm_feature = norm_feature
        self.fc1 = nn.Linear(768, embedding_dim*2)
        self.fc2 = nn.Linear(embedding_dim*2, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, num_classes)
        self.to_add = to_add_dim
        self.ortholinear_p = orthogonal(
            nn.Linear(embedding_dim, embedding_dim, bias=False))
        self.ortholinear_old = orthogonal(
            nn.Linear(embedding_dim, to_add_dim, bias=False))
        self.ortholinear_old2 = orthogonal(
            nn.Linear(embedding_dim, embedding_dim, bias=False))
        self.C = C

    def forward(self, x):
        """
        A forward pass through the model.
        Returns the classification logits, the learnt representation
            to match the old representation, the learnt representation
            to match the new' representation, and the final new representation
        """
        #         self.z.data = torch.clamp(self.z.data, max = 1)
        #         self.z = nn.Parameter(torch.clamp(self.z, max = 1))
        x = self.transformer(x).pooler_output
        features = self.fc1(x)
        old_features = self.fc2(features)
        features = features.reshape(features.size(0), -1)[:,
                                                          :self.embedding_dim]
        old_features = old_features[:, :self.to_add]
        if self.norm_feature:
            old_features = F.normalize(old_features)
            features = F.normalize(features)
        # old_features, features = features, old_features

        new_feature = torch.cat([old_features,
                                 self.ortholinear_p(self.C*features)], dim=1)
        bct_feature = self.ortholinear_old2(new_feature[:, :-self.to_add])
        to_add_feature = new_feature[:, -self.to_add:]

        new_feature = torch.cat([bct_feature, to_add_feature], dim=1)
        output = self.fc_out(features)
        return output, bct_feature, features, new_feature


def VIT_google_ortho_shallow(num_classes: int,
                             embedding_dim: int,
                             norm_feature: bool = True,
                             to_add_dim: int = 32,
                             C: float = 3,
                             **kwargs) -> nn.Module:
    """Get a Vit google wideshallow model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet50 Model.
    """
    return VIT_ortho_shallow(
        num_classes=num_classes,
        pretrained_name='google/vit-base-patch16-224-in21k',
        embedding_dim=embedding_dim,
        norm_feature=norm_feature,
        to_add_dim=to_add_dim,
        C=C,
    )
