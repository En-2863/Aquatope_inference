#### This repo is adapted from AQUATOPE https://github.com/zzhou612/aquatope project.

```bash
.
├── model_artifacts/                    # folder for saving the models
│   ├── lstm_encoder_decoder_{trace_id}.pt
│   └── predict_{trace_id}.pt
├── models/                             # files for building the model architecture
├── data.py                             # load the trace and build datasets
├── train_lstm_encoder_decoder.py       # train the encoder model
├── train_prediction_network.py         # train the prediction model
├── train.py                            # pipeline for gathering the whole training process
├── inference.py                        # pipeline for inference
└── visualize.py                        # plot the prediction and truth trace
```

To use it, first you need to download the Azure Functions Trace 2019 from https://github.com/Azure/AzurePublicDataset and place it under the project folder.

Training: 
```bash
python train.py --trace_id trace_id --dataset_dir ./
```
You can also train encoder and predcition network seperately, each of which has more detailed training args, including input_steps and out_steps.

Inference: 
```bash
python inference.py --trace trace_id --trace_file_path ./
```