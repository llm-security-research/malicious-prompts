import torch
import os
dependencies = ['torch']

def promptsentinel_unbalanced_v1(*args, **kwargs):
    # model = torch.load("./PromptSentinel/PromptSentinel-Unbalanced-v1/model.pth")
    model_path = os.path.join(os.path.dirname(__file__), 'PromptSentinel/PromptSentinel-Unbalanced-v1/model.pth')
    model = torch.load(model_path)

    return model
