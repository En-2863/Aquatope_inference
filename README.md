#### This repo is adapted from AQUATOPE https://github.com/zzhou612/aquatope project.

```bash
.
├── model_artifacts/                    # folder for saving the models
│   ├── lstm_encoder_decoder.pt
│   └── predict.pt
├── models/                             # files for building the model architecture
├── server.py                           # Aquatope inference server
├── train.py                            # pipeline for gathering the whole training process
├── data.py                             # load the trace and build datasets
├── train_lstm_encoder_decoder.py       # train the encoder model
├── train_prediction_network.py         # train the prediction model
├── inference.py                        # pipeline for inference (pending...)
└── visualize.py                        # plot the prediction and truth trace (pending...)
```

Training: 
```bash
python train.py --trace_file 'trace path relative to this train.py file' \
--model_path 'relative dir path to save the model' --input_length --output_length --period --interval \
--encoder_lr --encoder_dropout_p --predict_lr --predict_dropout_p
```
The value of this arguments may vary according to your need, and we also provide default value for
this arguments, except trace_file and model_path.

Also, for period, interval: Aquatope supposes traces to be periodic, with period and interval(inside a period), for example, period=7, interval=24 means a week with 24 hours in a day.
Here we set period=10, interval=100.

Serve: 
```bash
python server.py --period --interval --input_length --output_length --model_path
```
Here model_path refers to the path of predict.pt.