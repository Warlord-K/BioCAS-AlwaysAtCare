import os
from Preprocessing import preprocess
from models import Model
import json
import argparse
import matplotlib.pyplot as plt

"""
& C:/Users/Yatharth/AppData/Local/Programs/Python/Python39/python.exe "e:/Yatharth/C++ Files/Biocas/workingfiles/main.py" -t event -w "E:\Yatharth\C++ Files\Biocas\SPRSound-1.0\wav" -o "E:\Yatharth\C++ Files\Biocas\SPRSound-1.0\testcase\output\output.json"
"""

def model(args):
    #path = r"E:\Yatharth\C++ Files\Biocas\SPRSound-1.0\testcase\task2_wav"
    #out_path = "E:\Yatharth\C++ Files\Biocas\SPRSound-1.0\testcase\output\output.json"
    path = args.wav
    task = args.task
    out_path = args.out
    sounds = [ f"{path}{sound}" for sound in os.listdir(path) if sound.endswith(".wav")]
    images = list(map(preprocess,sounds))
    model = Model()
    pred = model.predict(images,task)
    taskoutput = {sounds[i][len(path)+1:]:pred[i] for i in range(len(pred))}
    with open(out_path,"w") as file:
            json.dump(taskoutput,file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Model")
    parser.add_argument('-t', '--task',default = 'event', type=int, help='Task to perform')
    parser.add_argument('-w', '--wav',default = './data/wav/1.wav', type=str, help='Path to wav files')
    parser.add_argument('-o', '--out',default = "./output.json", type=str, help='Path to output json')
    args = parser.parse_args()
    model(args)
    # images = model(args)
    # for img in images:
    #     plt.imshow(img)
    #     plt.show()

