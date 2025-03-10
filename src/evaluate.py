import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import BertTokenizer
from model import GitHubIssueClassifier
from time import gmtime, strftime

# Configuration d'inférence
CONFIG = {
    "max_length": 512,      
    "output_dim": 10,                
    "batch_size": 32,
    "test_data": "data/test_data.pkl",
    "model_save_path": "models/bert_test/07032025_081637/",
    "model_name": "bert_model_test_best.pth",
    "pretrained_model_name": "bert-base-uncased"
}

# Liste des noms de labels, dans l'ordre attendu
target_names = [
    "bug", "documentation", "duplicate", "test", "good first issue", 
    "help wanted", "invalid", "question", "refactoring", "wontfix"
]

# Initialisation du tokenizer BERT
tokenizer = BertTokenizer.from_pretrained(CONFIG["pretrained_model_name"])

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, num_labels=CONFIG["output_dim"]):
        self.num_labels = num_labels
        self.texts = df['body'].tolist()
        self.labels = df['labels_numeric'].apply(self._convert_label).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _convert_label(self, label):
        # Convertit la liste d'indices en un vecteur binaire
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

def evaluate(model, dataloader, criterion, device):
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
    avg_loss = running_loss / len(dataloader.dataset)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    preds = (all_outputs >= 0.5).astype(int)
    report = classification_report(all_targets, preds, target_names=target_names, zero_division=0)
    return avg_loss, report

def load_model_state(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chargement du dataset de test
    test_df = pd.read_pickle(CONFIG["test_data"])
    test_dataset = TextDataset(test_df, tokenizer, max_length=CONFIG["max_length"], num_labels=CONFIG["output_dim"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=16, pin_memory=True)
    
    # Initialisation du modèle BERT pour la classification multi-label
    model = GitHubIssueClassifier(num_labels=CONFIG["output_dim"], pretrained_model_name=CONFIG["pretrained_model_name"])
    
    # Charger les poids sauvegardés
    model_path = os.path.join(CONFIG["model_save_path"], CONFIG["model_name"])
    if not os.path.exists(model_path):
        print(f"Erreur : le modèle n'a pas été trouvé à {model_path}")
        return
        
    new_state_dict = load_model_state(model_path, device)
    model.load_state_dict(new_state_dict)

    model.to(device)
    
    print("Modèle chargé et prêt pour l'évaluation.")
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Évaluer le modèle sur le dataset de test
    test_loss, test_report = evaluate(model, test_loader, criterion, device)
    print("Test Loss :", test_loss)
    print("Rapport de classification sur le set de test :")
    print(test_report)

if __name__ == "__main__":
    main()
