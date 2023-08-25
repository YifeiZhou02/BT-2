#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Dict
from argparse import ArgumentParser

import torch
import yaml
from utils.getters import get_model
import json

from dataset import SubImageFolder
from utils.eval_utils import cmc_evaluate
from accelerate import Accelerator
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor


def main(config: Dict) -> None:
    """Run evaluation.

    :param config: A dictionary with all configurations to run evaluation.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    # Load models:
    if config.get('arch_params') is not None:
        model = get_model(config.get('arch_params'))
        checkpoint = torch.load(config.get('query_model_path'))
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state_dict'])
        query_model = model.to(device)
    else:
        query_model = torch.jit.load(config.get('query_model_path')).to(device)
    gallery_model = query_model
    if config.get('gallery_arch_params') is not None:
        gallery_model = get_model(config.get('gallery_arch_params'))
        checkpoint = torch.load(config.get('gallery_model_path'))
        gallery_model.load_state_dict(checkpoint['model_state_dict'])
    if config.get('gallery_model_path') is not None:
        gallery_model = torch.jit.load(
            config.get('gallery_model_path')).to(device)

    if isinstance(gallery_model, torch.nn.DataParallel):
        gallery_model = gallery_model.module
    if isinstance(query_model, torch.nn.DataParallel):
        query_model = query_model.module
    data = SubImageFolder(**config.get('dataset_params'))

    # this part is for multimodal experiments
    path2caption_path = config.get('path2caption_path')
    if path2caption_path is not None:
        with open(path2caption_path, 'r') as f:
            path2caption = json.load(f)
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32")
        vit_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        gallery_model = (tokenizer, path2caption, processor, vit_model)

    val_loader = [value for i, value in enumerate(
        data.val_loader) if i % 1 == 0]

#     print(model.module.z)

    cmc_out, mean_ap_out = cmc_evaluate(
        gallery_model,
        query_model,
        val_loader,
        device,
        **config.get('eval_params')
    )

    if config.get('eval_params').get('per_class'):
        print('CMC Top-1 = {}, CMC Top-5 = {}'.format(*cmc_out[0]))
        print('Per class CMC: {}'.format(cmc_out[1]))
    else:
        print('CMC Top-1 = {}, CMC Top-5 = {}'.format(*cmc_out))

    if mean_ap_out is not None:
        print('mAP = {}'.format(mean_ap_out))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for this pipeline.')
    args = parser.parse_args()
    with open(args.config) as f:
        read_config = yaml.safe_load(f)
    main(read_config)
