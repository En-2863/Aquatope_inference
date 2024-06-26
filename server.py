import sys
import json
import select
import time
import os
import torch
import argparse
import numpy as np
from inference import inference
from utils import get_device, time_emdedding

current_dir = os.path.dirname(os.path.abspath(__file__))

class AquatopeServer():
    def __init__(self, period, interval, input_length, 
                 output_length=1, model_path=None):
        self.model = None
        self.model_path = model_path
        self.period = period
        self.interval = interval
        self.time_length = period * interval
        self.input_length = input_length
        self.output_length = output_length
        
        time_embedding = time_emdedding(period, interval, self.time_length)
        self.time_embedding= np.concatenate([time_embedding, time_embedding], axis=0)
        self.device = get_device()
        
        if self.model_path is not None:
            self.model = self.load_model(self.model_path, self.device)
                    
    def load_model(self, model_path: str, device: str):
        model_path = os.path.join(current_dir, model_path)
        predict_loc = os.path.join(current_dir, model_path)
        predict = torch.load(predict_loc, map_location=device).eval()
        return predict.to(device)
    
    def get_time_embedding(self, cur_time):
        cur_id = cur_time % self.time_length
        return self.time_embedding[cur_id:cur_id+self.input_length]
        
    def run(self):
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 10)
            
            if ready:
                line = sys.stdin.readline().strip()
                content = json.loads(line)
                
                try:
                    content = json.loads('{"model_path": "./model_artifacts/predict.pt", "input_data": [5148, 3214, 2973, 2973, 10000, 6165, 2433, 5312, 4299, 465, 1444, 4530, 9388, 2994, 2276, 3901, 2162, 2834, 3547, 1377, 6387, 1151, 1017, 4159, 6050, 4082, 3273, 2812, 2452, 7393, 4562, 2618, 2015, 5565, 7276, 6842, 1723, 2793, 2412, 3093, 6347, 8822, 4685, 2075, 3486, 9920, 3832, 2817], "cur_time": 0}')
                    model_path = content.get("model_path", None)
                    input_data = content.get("input_data", None)
                    cur_time = content.get("cur_time", None)
                    
                    # load model if model path is provided
                    if model_path is not None and self.model_path != model_path:
                        self.model = self.load_model(model_path, self.device)
                        self.model_path = model_path
                        
                    # sanity check
                    if self.model is None:
                        raise ValueError("Model is not loaded")
                    
                    if input_data is None or cur_time is None:
                        raise ValueError("Input data and cur time are required")
                    
                    time_emdedding = self.get_time_embedding(cur_time)
                    input_data = np.array(input_data)
                    trace_mu = np.mean(input_data)
                    trace_sigma = np.std(input_data)
                    input_data = (input_data - trace_mu) / trace_sigma

                    trace = np.concatenate([input_data[:, np.newaxis], time_emdedding], axis=1)
                    mean, var = inference(trace, external=[0, 0, 0, 0], model=self.model)
                    mean = int(- mean * trace_sigma + trace_mu)
                    
                    # write output to stdout
                    print(mean)
                     
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aquatope Inferencing Server")
    parser.add_argument("--period", action="store", type=int, default=10)
    parser.add_argument("--interval", action="store", type=int, default=100)
    parser.add_argument("--input_length", action="store", type=int, default=48)
    parser.add_argument("--output_length", action="store", type=int, default=1)
    parser.add_argument("--model_path", action="store", type=str, default=None)
    
    args = parser.parse_args()
    period = args.period
    interval = args.interval
    input_length = args.input_length
    output_length = args.output_length
    model_path = args.model_path
    server = AquatopeServer(period=period, interval=interval, input_length=input_length,
                            output_length=output_length, model_path=model_path)
    server.run()