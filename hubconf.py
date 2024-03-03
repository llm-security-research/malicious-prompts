import torch
dependencies = ['torch']

def promptsentinel_unbalanced_v1(*args, **kwargs):
    model = torch.load("./PromptSentinel/PromptSentinel-Unbalanced-v1/model.pth")

    return model
