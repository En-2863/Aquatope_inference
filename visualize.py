import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import load_trace

def visualize(trace, trace_predict, trace_id=None):
    plt.plot(trace, label="True")
    plt.plot(trace_predict, label="Predict")
    plt.xlabel('Time')
    plt.ylabel('Invocation Rate')
    plt.legend()
    plt.savefig(f'trace_predict-{trace_id}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train prediction network")
    parser.add_argument("--trace_file_path", type=str, default='./')
    parser.add_argument("--trace_id", action="store", type=str, default='6c871093253858350d730f80cfbc5109aa2529d9cda37341989ea8854485663e')

    args = parser.parse_args()
    trace_file_path = args.trace_file_path
    function_id = args.trace_id
    
    trace = load_trace(trace_file_path, function_id)
    trace_predict = np.load(f'trace_predict-{function_id}.npy')
    visualize(trace, trace_predict, function_id)