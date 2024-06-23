import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))
import data
from models.encoder_decoder_dropout import *
from train_lstm_encoder_decoder import lstm_encoder_decoder
from train_prediction_network import prediction_network

import utils


def train(trace_id=None,
          dataset_dir=None,
          model_artifacts_dir=None):
    # Train LSTM encoder decoder
    if trace_id is None:
        raise ValueError("trace_id is required")
    
    encoder = lstm_encoder_decoder(trace_id=trace_id, dataset_dir=dataset_dir, 
                         model_artifacts_dir=model_artifacts_dir)
    
    # Train prediction network
    predict_network = prediction_network(trace_id=trace_id, dataset_dir=dataset_dir, 
                         model_artifacts_dir=model_artifacts_dir, encoder_decoder=encoder)
    
    return predict_network


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Train LSTM encoder decoder")
    parser.add_argument("--trace_id", action="store", type=str, default='6c871093253858350d730f80cfbc5109aa2529d9cda37341989ea8854485663e')
    parser.add_argument("--dataset_dir", action="store", type=str, default='./')
    
    args = parser.parse_args()
    trace_id = args.trace_id
    dataset_dir = args.dataset_dir
    model_artifacts_dir = SCHED_DIR / "model_artifacts"

    train(trace_id, dataset_dir, model_artifacts_dir)