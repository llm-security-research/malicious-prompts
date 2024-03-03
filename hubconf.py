import torch
import os
dependencies = ['torch']

def promptsentinel_unbalanced_v1(*args, **kwargs):
    model_path = os.path.join(os.path.dirname(__file__), 'PromptSentinel/PromptSentinel-Unbalanced-v1/model.pth')
    model = torch.load(model_path)

    return model

def promptsentinel_balanced_v1(*args, **kwargs):
    model_path = os.path.join(os.path.dirname(__file__), 'PromptSentinel/PromptSentinel-Balanced-v1/model.pth')
    model = torch.load(model_path)

    return model

def promptsentinel_unbalanced_paraphrase_v1(*args, **kwargs):
    model_path = os.path.join(os.path.dirname(__file__), 'PromptSentinel/PromptSentinel-Unbalanced-Paraphrase-v1/model.pth')
    model = torch.load(model_path)

    return model