import argparse
from model import AudioClassifier
import json
import torch
from audio import AudioUtil

def predict(audio_file):
    mymodel = AudioClassifier()
    mymodel.load_state_dict(torch.load("./model_state_dict.pt"))
    aud = AudioUtil.open(audio_file)
    dur_aud = AudioUtil.pad_trunc(aud, 10000)
    shift_aud = AudioUtil.time_shift(dur_aud, 0.4)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    aug_sgram = aug_sgram.unsqueeze_(0)
    output = mymodel(aug_sgram)
    prediction = "Adventitious" if output > 0.5 else "Normal"
    return prediction

def model(args):
    file = args.wav
    task = args.task
    input_json = args.input_json
    output_json = args.output_json


    if task == 'record':
        if input_json is not None:
            with open(input_json,'r') as f:
                data1 = json.load(f)
        
        data1["record_annotation"] = predict(file)
        
        with open(output_json,'w') as f:
            json.dump(data1,f)
        
    elif task == 'event':
        print("Still Working On It")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Model")
    parser.add_argument('-w', '--wav',default = './data/wav/1.wav', type=str, help='Path to wav file')
    parser.add_argument('-t', '--task',default = 'record', type=str, help='Task to perform')
    parser.add_argument('-i', '--input_json',default = None, type=str, help='Path to input json')
    parser.add_argument('-o', '--output_json',default = "./output.json", type=str, help='Path to output json')
    args = parser.parse_args()
    model(args)


