import os
from pathlib import Path
import numpy as np
import torch
import data
import utils
from models.predict import *

current_dir = os.path.dirname(os.path.abspath(__file__))

def prediction_network(trace_file=None, model_path=None,
                       num_epochs=128, input_length=48, output_length=1, 
                       batch_size=128, period=10, interval=10,
                       learning_rate=1e-2, 
                       dropout_p=0.2,
                       encoder_decoder=None):
    # Load datasets
    trace_file = os.path.join(current_dir, trace_file)
    with open(trace_file, 'r') as f:
        trace = f.read()
        trace = trace.split('\n')
        trace = [int(x) for x in trace if x]
    trace = np.array(trace).flatten()
    df, split_dfs, samples = data.pipeline(
        n_input_steps=input_length,
        n_pred_steps=output_length,
        trace=trace,
        period=period,
        interval=interval
    )
    datasets = data.get_datasets(
        samples=samples, n_input_steps=input_length, pretraining=False
    )

    # Train LSTM encoder decoder
    device = utils.get_device()
    
    if encoder_decoder is None:
        encoder_decoder_loc = os.path.join(model_path, "lstm_encoder_decoder.pt")
        encoder_decoder = torch.load(encoder_decoder_loc)
    
    prediction_network = Predict(
        n_extracted_features=input_length,
        n_external_features=4,
        n_output_steps=output_length,
        p=dropout_p,
        encoder_decoder=encoder_decoder,
    )

    model, losses = utils.train_prediction_network(
        device=device,
        datasets=datasets,
        prediction_network=prediction_network,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_tqdm=True,
    )
    

    model_name = "predict"
    utils.save(model, name=model_name, path=model_path)
    return model


#if __name__ == "__main__":
#    
#    parser = argparse.ArgumentParser(description="Train prediction network")
#    parser.add_argument("--n_input_steps", action="store", type=int, default=48)
#    parser.add_argument("--n_output_steps", action="store", type=int, default=1)
#    parser.add_argument("--num_days", action="store", type=int, default=7)
#    parser.add_argument("--num_epochs", action="store", type=int, default=128)
#    parser.add_argument("--batch_size", action="store", type=int, default=128)
#    parser.add_argument("--learning_rate", action="store", type=float, default=1e-2)
#    parser.add_argument("--dropout_p", action="store", type=float, default=0.25)
#    parser.add_argument("--trace_id", action="store", type=str, default='6c871093253858350d730f80cfbc5109aa2529d9cda37341989ea8854485663e')
#    parser.add_argument("--dataset_dir", action="store", type=str, default='./')
#    
#    args = parser.parse_args()
#    n_input_steps = args.n_input_steps
#    n_output_steps = args.n_output_steps
#    num_days = args.num_days
#    trace_id = args.trace_id
#    dataset_dir = args.dataset_dir
#    model_artifacts_dir = SCHED_DIR / "model_artifacts"
#    num_epochs = args.num_epochs
#    batch_size = args.batch_size
#    learning_rate = args.learning_rate
#    dropout_p = args.dropout_p
#    
#    encoder_decoder_loc = model_artifacts_dir / "lstm_encoder_decoder.pt"
#    encoder_decoder = torch.load(encoder_decoder_loc)
#    
#    prediction_network(n_input_steps, n_output_steps, num_days, trace_id, 
#                       dataset_dir, model_artifacts_dir, num_epochs, 
#                       batch_size, learning_rate, dropout_p, encoder_decoder)
