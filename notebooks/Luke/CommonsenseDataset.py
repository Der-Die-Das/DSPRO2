import os
import pandas as pd

from torch.utils.data import Dataset
import random

class CommonsenseDataset(Dataset):
  def __init__(self, data_path):
    super().__init__()
    
    if not os.path.exists(data_path):
      raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    self.data = pd.read_parquet(data_path)
    
    self.data = self.data[["question", "choices.text", "choices.label", "answerKey"]]
    
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    item = self.data.iloc[idx]
    choices = item["choices.text"]
    labels = item["choices.label"]
    
    if len(labels) == len(choices):
      ziped = list(zip(choices, labels))
      random.shuffle(ziped)
      choices, labels = zip(*ziped)
    
    answer_key = item["answerKey"]

    return {
        "question": item["question"],
        "choices": [choice for choice in choices],
        "labels": [lable == answer_key for lable in labels],
    }