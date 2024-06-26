import os
import numpy as np
import data
from models.encoder_decoder_dropout import *
import utils

current_dir = os.path.dirname(os.path.abspath(__file__))

def lstm_encoder_decoder(trace_file=None, model_path=None, 
                         num_epochs=128, input_length=48, output_length=1,
                         batch_size=128, period=10, interval=10,
                         learning_rate=1e-2, 
                         variational_dropout_p=0.25):
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
        samples=samples, n_input_steps=input_length, pretraining=True
    )
    encoder_in_features = datasets["train"].X.shape[-1]  # 5
    device = utils.get_device()

    # Train LSTM encoder decoder
    model = VDEncoderDecoder(
        in_features=encoder_in_features,
        input_steps=input_length,
        output_steps=output_length,
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

    model_name = "lstm_encoder_decoder"
    utils.save(model, name=model_name, path=model_path)
    
    return model


# if __name__ == "__main__":
#     # Parse args
#     parser = argparse.ArgumentParser(description="Train LSTM encoder decoder")
#     parser.add_argument("--n_input_steps", action="store", type=int, default=48)
#     parser.add_argument("--n_output_steps", action="store", type=int, default=12)
#     parser.add_argument("--num_days", action="store", type=int, default=7)
#     parser.add_argument("--num_epochs", action="store", type=int, default=128)
#     parser.add_argument("--batch_size", action="store", type=int, default=128)
#     parser.add_argument("--learning_rate", action="store", type=float, default=1e-2)
#     parser.add_argument("--variational_dropout_p", action="store", type=float, default=0.25)
#     parser.add_argument("--trace_id", action="store", type=str, default='6c871093253858350d730f80cfbc5109aa2529d9cda37341989ea8854485663e')
#     parser.add_argument("--dataset_dir", action="store", type=str, default='./')
# 
#     args = parser.parse_args()
#     n_input_steps = args.n_input_steps
#     n_output_steps = args.n_output_steps
#     num_days = args.num_days
#     trace_id = args.trace_id
#     dataset_dir = args.dataset_dir
#     model_artifacts_dir = SCHED_DIR / "model_artifacts"
#     num_epochs = args.num_epochs
#     batch_size = args.batch_size
#     learning_rate = args.learning_rate
#     variational_dropout_p = args.variational_dropout_p
#     
#     lstm_encoder_decoder(n_input_steps, n_output_steps, num_days, trace_id, 
#                          dataset_dir, model_artifacts_dir, num_epochs, 
#                          batch_size, learning_rate, variational_dropout_p)
