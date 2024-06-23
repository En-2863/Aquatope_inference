import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))

import models.variational_dropout as vd
from models.predict import *

import data
import utils

cpu = lambda x: x.cpu().detach().numpy()

MODEL = None
MODEL_ARTIFACTS_DIR = SCHED_DIR / "model_artifacts"


def load_trained_model(model_artifacts_dir: str, device: str, model="predict.pt"):
    predict_loc = os.path.join(model_artifacts_dir, model)
    predict = torch.load(predict_loc, map_location=device).eval()
    return predict.to(device)


def dropout_on(m: nn.Module):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.train()


def dropout_off(m: nn.Module):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.eval()


def inference(x: list, 
              external: list, 
              mc_dropout: bool = False, 
              batch_size: int = 1,
              model=None):
    device = utils.get_device()
    x = np.expand_dims(x, axis=0)
    external = np.expand_dims(external, axis=0)
    x = torch.tensor(np.array(x, dtype=np.float32), device=device)
    external = torch.tensor(np.array(external, dtype=np.float32), device=device)

    res = []
    for _ in range(batch_size):
        res.append(model((x, external)).to(device).item())
    mean = np.mean(res)
    var = np.var(res)
    return mean, var


def trace_inference(trace=None,
                    trace_id=None,
                    external=None, 
                    n_input_steps=48,
                    n_output_steps=1, 
                    period=7, 
                    interval=24, 
                    days=7,
                    batch_size=1,
                    predict_model=None):
    model_artifacts_dir = SCHED_DIR / "model_artifacts" 
    device = utils.get_device()
    
    if predict_model is None:
        predict = load_trained_model(model_artifacts_dir=model_artifacts_dir, 
                                     device=device, model=f"predict-{trace_id}.pt")
    else:
        predict = predict_model
        
    predict = predict.apply(dropout_off)
    predict = predict.to(device)
    
    max_time_length = trace.shape[0]
    trace_mu = np.mean(trace)
    trace_sigma = np.std(trace)
    trace = (trace - trace_mu) / trace_sigma
    trace_with_time_embedding = utils.time_emdedding(trace, period, interval, days)
    
    time_steps = max_time_length - n_input_steps
    trace_predict = trace_with_time_embedding.copy()
    
    # external = np.array([0, 0, 0, 0])
    # using tqdm
    for i in tqdm(range(time_steps)):
        x = trace_with_time_embedding[i:n_input_steps+i]
        mean, var = inference(x=x, external=external, mc_dropout=False, 
                              batch_size=batch_size, model=predict)
        trace_predict[i + n_input_steps, 0] = - mean * trace_sigma + trace_mu
        
    return trace_predict[:, 0]


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Train prediction network")
    parser.add_argument("--trace_file_path", type=str, default='./')
    parser.add_argument("--n_input_steps", action="store", type=int, default=48)
    parser.add_argument("--n_output_steps", action="store", type=int, default=1)
    parser.add_argument("--trace_id", action="store", type=str, default='6c871093253858350d730f80cfbc5109aa2529d9cda37341989ea8854485663e')
    # 6c871093253858350d730f80cfbc5109aa2529d9cda37341989ea8854485663e
    args = parser.parse_args()
    trace_file_path = args.trace_file_path
    n_input_steps = args.n_input_steps
    n_output_steps = args.n_output_steps
    function_id = args.trace_id
    
    trace = data.load_trace(trace_file_path, function_id)
    external = [0, 0, 0, 0]
    trace_predict = trace_inference(trace=trace,
                                    trace_id=function_id, external=external,
                                    n_input_steps=n_input_steps, 
                                    n_output_steps=n_output_steps)

    # save trace_predict
    np.save(f'trace_predict-{function_id}.npy', trace_predict)