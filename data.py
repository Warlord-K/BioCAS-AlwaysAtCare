import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
def load():
    df = pd.read_csv('data.csv')
    df1 = pd.DataFrame()
    path = "./SPRSound/json/"
    files = df["filename"]
    start = []
    end = []
    types = []
    filenames = []
    record_annotations = []
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(path, file), 'r') as f:
                data = json.load(f)
                events = list(data["event_annotation"])
                
                if len(events) == 0:
                    filenames.append(file)
                    record_annotations.append(data["record_annotation"])
                    start.append("None")
                    end.append("None")
                    types.append("None")
                else:
                    for event in events:
                        start.append(event["start"])
                        end.append(event["end"])
                        types.append(event["type"])
                        filenames.append(file)
                        record_annotations.append(data["record_annotation"])
    df1["filename"] = filenames
    df1["record_annotation"] = record_annotations
    df1["start"] = start
    df1["end"] = end
    df1["type"] = types
    df1.to_csv("data_with_annotations.csv",index = False)
    print("done")

def encode(df):
    le = LabelEncoder()
    encoded = le.fit_transform(df["record_annotation"])
    df["encoded_record_annotations"] = encoded
    return df

def binary_data(df):
    df = df[df["encoded_record_annotations"] != 4]
    df["binary_record_annotations"] = df["encoded_record_annotations"].apply(lambda x: 0 if x == 3 else 1)
    return df

df = binary_data(pd.read_csv("data_encoded.csv"))
df.to_csv("data_encoded_new.csv",index = False)


"""
Used Code

df["filename"] = files
df["record_annotation"] = record_annotations
df.to_csv("data.csv",index=False)

"""