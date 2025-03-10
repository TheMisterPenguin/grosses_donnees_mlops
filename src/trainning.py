
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from time import gmtime, strftime
import wandb
from transformers import BertTokenizer
from model import GitHubIssueClassifier

# Hyperparamètres et configuration
CONFIG = {
    "max_length": 512,
    "output_dim": 10,
    "learning_rate": 2e-05,
    "batch_size": 32,
    "num_epochs": 5,
    "train_data": "data/train_data.pkl",  
    "val_data": "data/dev_data.pkl",
    "test_data": "data/test_data.pkl",
    "model_save_path": f"models/bert_test/{strftime('%d%m%Y_%H%M%S', gmtime())}/",
    "model_name": "bert_model_test.pth",
    "pretrained_model_name": "bert-base-uncased"
}

if not os.path.isdir(CONFIG["model_save_path"]):
    os.makedirs(CONFIG["model_save_path"])

target_names = ["bug", "documentation", "duplicate", "test", "good first issue", 
                "help wanted", "invalid", "question", "refactoring", "wontfix"]

# Initialisation du tokenizer BERT
tokenizer = BertTokenizer.from_pretrained(CONFIG["pretrained_model_name"])

class TextDataset(Dataset):

    def __init__(self, df, tokenizer, max_length=128, num_labels=CONFIG["output_dim"]):
        self.num_labels = num_labels
        self.texts = df['body'].tolist()
        self.labels = df['labels_numeric'].apply(self._convert_label).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _convert_label(self, label):
        # Convertit la liste d'indices en un vecteur binaire de dimension num_labels
        return torch.tensor([1 if i in label else 0 for i in range(self.num_labels)], dtype=torch.float32)
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0) 
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = self.labels[idx]
        return input_ids, attention_mask, label

# Fonction de calibration des seuils (identique à l'ancienne version)
def calibrate_thresholds(model, dataloader, device, num_labels=10):
    model.eval()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    best_thresholds = [0.5] * num_labels
    candidate_thresholds = np.linspace(0.5, 1.0, num=100)
    for label_idx in range(num_labels):
        best_f1 = 0.0
        best_t = 0.5
        for t in candidate_thresholds:
            preds_label = (all_outputs[:, label_idx] >= t).astype(int)
            f1 = f1_score(all_targets[:, label_idx], preds_label, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[label_idx] = best_t
    return best_thresholds

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        # Utilisation de l'autocast pour le calcul en précision mixte
        with autocast("cuda"):
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
        # Mise à l'échelle et rétropropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * input_ids.size(0)
    return running_loss / len(dataloader.dataset)

def val_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * input_ids.size(0)
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    val_loss = running_loss / len(dataloader.dataset)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    preds = (all_outputs >= 0.5).astype(int)
    f1_macro = f1_score(all_targets, preds, average='macro')
    return val_loss, f1_macro

def evaluate(model, dataloader, set_name, criterion, thresholds, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * input_ids.size(0)
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    val_loss = running_loss / len(dataloader.dataset)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    calibrated_outputs = np.zeros_like(all_outputs)
    for i in range(CONFIG["output_dim"]):
        calibrated_outputs[:, i] = (all_outputs[:, i] >= thresholds[i]).astype(int)
    report = classification_report(all_targets, calibrated_outputs, target_names=target_names, zero_division=0)
    print(f"\n\nReport {set_name} : \n{report}\n\n")
    return report