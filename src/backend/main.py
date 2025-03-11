from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
import numpy as np
from transformers import BertTokenizer, logging
from model_v2 import GitHubIssueClassifier  # Assurez-vous que ce module correspond à votre modèle BERT

# Réduire la verbosité de Transformers
logging.set_verbosity_error()

# Configuration d'inférence
CONFIG = {
    "max_length": 512,                # Longueur maximale pour tokeniser le texte
    "output_dim": 10,                 # Nombre de labels
    "model_save_path": "models/",     # Chemin vers le dossier contenant le modèle entraîné
    "model_name": "bert_model_test.pth",
    "pretrained_model_name": "bert-base-uncased"
}

# Liste des noms de labels dans l'ordre attendu
target_names = [
    "bug", "documentation", "duplicate", "test", "good first issue", 
    "help wanted", "invalid", "question", "wontfix"
]

# Initialisation du tokenizer BERT
tokenizer = BertTokenizer.from_pretrained(CONFIG["pretrained_model_name"])

def load_model_state(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

def predict_text(text, thresholds=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instanciation et chargement du modèle
    model = GitHubIssueClassifier(num_labels=CONFIG["output_dim"],pretrained_model_name=CONFIG["pretrained_model_name"])
    model_path = os.path.join(CONFIG["model_save_path"], CONFIG["model_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle sauvegardé n'a pas été trouvé : {model_path}")
    new_state_dict = load_model_state(model_path, device)
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Tokenisation du texte
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=CONFIG["max_length"],
        return_tensors='pt'
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.cpu().numpy()[0]
    
    # Utilisation de seuils prédéfinis
    thresholds = [np.float64(0.5), np.float64(0.5), np.float64(0.8636363636363636), np.float64(0.5), np.float64(0.5), np.float64(0.8383838383838385), np.float64(0.51010101010101), np.float64(0.5), np.float64(0.8737373737373737), np.float64(0.5)]
    preds = (logits >= np.array(thresholds)).astype(int)
    predicted_labels = [target_names[i] for i, pred in enumerate(preds) if pred == 1]
    return predicted_labels, logits.tolist()

# Modèles Pydantic pour la requête et la réponse
class IssueRequest(BaseModel):
    text: str

class IssueResponse(BaseModel):
    predicted_labels: list

app = FastAPI()

@app.post("/predict", response_model=IssueResponse)
def predict_issue(request: IssueRequest):
    if request.text is None or not request.text.strip():
        raise HTTPException(status_code=400, detail="Le texte de l'issue ne peut pas être vide.")
    
    try:
        labels, logits = predict_text(request.text)
        return IssueResponse(predicted_labels=labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)