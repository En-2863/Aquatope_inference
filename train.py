import argparse
import sys
from pathlib import Path
import os
import data
from models.encoder_decoder_dropout import *
from train_lstm_encoder_decoder import lstm_encoder_decoder
from train_prediction_network import prediction_network

import utils


def train(trace_file=None,
          model_path=None,
          epoch=128,
          input_length=48,
          output_length=1,
          period=10,
          interval=100,
          encoder_lr=1e-2,
          encoder_dropout_p=0.25,
          predict_lr=1e-2,
          predict_dropout_p=0.2):
    # Train LSTM encoder decoder
    encoder = lstm_encoder_decoder(trace_file, model_path, epoch, 
                                   input_length, output_length, period, interval,
                                   encoder_lr, encoder_dropout_p)
    
    # Train prediction network
    predict_network = prediction_network(trace_file, model_path, epoch,
                                         input_length, output_length, period, interval,
                                         predict_lr, predict_dropout_p, encoder)
    
    return predict_network

current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Train LSTM encoder decoder")
    parser.add_argument("--trace_file", action="store", type=str)
    parser.add_argument("--model_path", action="store", type=str)
    parser.add_argument("--epoch", action="store", type=int, default=128)
    parser.add_argument("--input_length", action="store", type=int, default=48)
    parser.add_argument("--output_length", action="store", type=int, default=1)
    parser.add_argument("--period", action="store", type=int, default=10)
    parser.add_argument("--interval", action="store", type=int, default=100)
    parser.add_argument("--encoder_lr", action="store", type=float, default=1e-2)
    parser.add_argument("--encoder_dropout_p", action="store", type=float, default=0.25)
    parser.add_argument("--predict_lr", action="store", type=float, default=1e-2)
    parser.add_argument("--predict_dropout_p", action="store", type=float, default=0.2)
    
    args = parser.parse_args()
    trace_file = args.trace_file
    input_length = args.input_length
    epoch = args.epoch
    output_length = args.output_length
    period = args.period
    interval = args.interval
    model_path = args.model_path
    encoder_lr = args.encoder_lr
    encoder_dropout_p = args.encoder_dropout_p
    predict_lr = args.predict_lr
    predict_dropout_p = args.predict_dropout_p

    train(trace_file, model_path, epoch, input_length, output_length, period, interval,
          encoder_lr, encoder_dropout_p, predict_lr, predict_dropout_p)