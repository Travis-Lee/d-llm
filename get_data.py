import urllib.request
import tiktoken
import zipfile
import os
from pathlib import Path
import pandas as pd
import torch 
from torch.utils.data import DataLoader,Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [                                      #A
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [                                  #B
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [                                      #C
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    with urllib.request.urlopen(url) as response:          #A
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:        #B
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)          #C
    print(f"File downloaded and saved as {data_file_path}")

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]                                 #A
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)      #B
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])              #C
    return balanced_df

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)     #A

    train_end = int(len(df) * train_frac)                               #B
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]                                           #C
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

def prepare_dataset():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "DataSet"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    #print("df:",df)
    #print("label:",df["Label"].value_counts())
    
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)  #D
    
    train_df.to_csv("DataSet/train.csv", index=None)
    validation_df.to_csv("DataSet/validation.csv", index=None)
    test_df.to_csv("DataSet/test.csv", index=None)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    
    train_dataset = SpamDataset(
        csv_file="DataSet/train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    print(train_dataset.max_length)
    
    val_dataset = SpamDataset(
        csv_file="DataSet/validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    print(val_dataset.max_length)
    
    test_dataset = SpamDataset(
        csv_file="DataSet/test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    print(test_dataset.max_length)
    num_workers = 0                  #A
    batch_size = 8
    torch.manual_seed(123)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


