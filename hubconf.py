import torch
import os
dependencies = ['torch']

from prediction_pipeline import *

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

def prediction_pipeline(*args, **kwargs):
    valid_strings = ['PromptSentinel-Unbalanced-v1', 'PromptSentinel-Balanced-v1', 'PromptSentinel-Unbalanced-Paraphrase-v1']
    arg1 = args[0] if args and args[0] in valid_strings else 'PromptSentinel-Unbalanced-Paraphrase-v1'
    model_path = os.path.join(os.path.dirname(__file__), f'PromptSentinel/{arg1}/model.pth')

    model = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path, map_location=torch.device('cpu'))

    pipeline = PredictionPipeline(model)

    return pipeline





