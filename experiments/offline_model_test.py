import asyncio
import json
import os
from functools import partial

import orjson
import torch
import time
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Offline
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader
)
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe,Pipeline, process_wrapper
from sae_auto_interp.scorers import FuzzingScorer, DetectionScorer

os.environ['HF_HOME'] = '/user/work/cp20141/repos/sae/experiments/cache'
#client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.8,max_model_len=5120, num_gpus=1)
client = Offline("google/gemma-2-9b",max_memory=0.8,max_model_len=4096, num_gpus=1)
