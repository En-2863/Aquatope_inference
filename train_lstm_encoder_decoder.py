import argparse
import sys
import uuid
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))
import data
from models.encoder_decoder_dropout import *

import utils


def lstm_encoder_decoder(n_input_steps=48, 
                         n_output_steps=12, 
                         num_days=7, 
                         trace_id=None, 
                         dataset_dir=None, 
                         model_artifacts_dir=None, 
                         num_epochs=128, 
                         batch_size=128, 
                         learning_rate=1e-2, 
                         variational_dropout_p=0.25):
    # Load datasets
    df, split_dfs, samples = data.pipeline(
        n_input_steps=n_input_steps,
        n_pred_steps=n_output_steps,
        hash_function=trace_id,
        dataset_dir=dataset_dir,
        num_days=num_days,
    )
    datasets = data.get_datasets(
        samples=samples, n_input_steps=n_input_steps, pretraining=True
    )
    encoder_in_features = datasets["train"].X.shape[-1]  # 5
    device = utils.get_device()

    # Train LSTM encoder decoder
    model = VDEncoderDecoder(
        in_features=encoder_in_features,
        input_steps=n_input_steps,
        output_steps=n_output_steps,
        p=variational_dropout_p,
    )
    model, losses = utils.train_encoder_decoder(
        device=device,
        model=model,
        datasets=datasets,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_tqdm=True,
    )
    if trace_id is None:
        model_name = f"lstm_encoder_decoder-{uuid.uuid4().hex}"
    else:
        model_name = f"lstm_encoder_decoder-{trace_id}"
    utils.save(model, name=model_name, path=model_artifacts_dir)
    
    return model


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Train LSTM encoder decoder")
    parser.add_argument("--n_input_steps", action="store", type=int, default=48)
    parser.add_argument("--n_output_steps", action="store", type=int, default=12)
    parser.add_argument("--num_days", action="store", type=int, default=7)
    parser.add_argument("--num_epochs", action="store", type=int, default=128)
    parser.add_argument("--batch_size", action="store", type=int, default=128)
    parser.add_argument("--learning_rate", action="store", type=float, default=1e-2)
    parser.add_argument("--variational_dropout_p", action="store", type=float, default=0.25)
    parser.add_argument("--trace_id", action="store", type=str, default='6c871093253858350d730f80cfbc5109aa2529d9cda37341989ea8854485663e')
    parser.add_argument("--dataset_dir", action="store", type=str, default='./')

    args = parser.parse_args()
    n_input_steps = args.n_input_steps
    n_output_steps = args.n_output_steps
    num_days = args.num_days
    trace_id = args.trace_id
    dataset_dir = args.dataset_dir
    model_artifacts_dir = SCHED_DIR / "model_artifacts"
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    variational_dropout_p = args.variational_dropout_p
    
    lstm_encoder_decoder(n_input_steps, n_output_steps, num_days, trace_id, 
                         dataset_dir, model_artifacts_dir, num_epochs, 
                         batch_size, learning_rate, variational_dropout_p)
