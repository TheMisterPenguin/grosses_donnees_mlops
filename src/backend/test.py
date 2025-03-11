import os
import torch
import numpy as np
from transformers import BertTokenizer, logging
from model_v2 import GitHubIssueClassifier


logging.set_verbosity_error()

def load_model_state(model_path, device):
    # Charger le state_dict sauvegardé
    state_dict = torch.load(model_path, map_location=device)
    # Créer un nouveau state_dict sans le préfixe "module."
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

# Configuration d'inférence
CONFIG = {
    "max_length": 512,                # Longueur maximale pour tokeniser le texte
    "output_dim": 10,                 # Nombre de labels
    "model_save_path": "models/",  # Mettez à jour ce chemin avec le dossier utilisé lors de l'entraînement
    "model_name": "bert_model_test.pth",
    "pretrained_model_name": "bert-base-uncased"
}

# Liste des noms de labels, dans l'ordre attendu (adapter selon vos labels)
target_names = ["bug", "documentation", "duplicate", "test", "good first issue", 
                "help wanted", "invalid", "question", "refactoring", "wontfix"]

# Initialisation du tokenizer
tokenizer = BertTokenizer.from_pretrained(CONFIG["pretrained_model_name"])

def predict_text(text, thresholds=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger le modèle BERT pour la classification multi-label
    model = GitHubIssueClassifier(num_labels=CONFIG["output_dim"],pretrained_model_name=CONFIG["pretrained_model_name"])
    model_path = os.path.join(CONFIG["model_save_path"], CONFIG["model_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle sauvegardé n'a pas été trouvé : {model_path}")
    
    # Charger le state_dict en supprimant le préfixe "module." s'il existe
    new_state_dict = load_model_state(model_path, device)
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Tokeniser le texte
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
    # Si aucun seuil n'est fourni, utiliser 0.5 par défaut pour tous les labels
    thresholds = [np.float64(0.5), np.float64(0.5), np.float64(0.5151515151515151), np.float64(0.51010101010101), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.797979797979798), np.float64(0.5050505050505051), np.float64(0.7777777777777778)]
    if thresholds is None:
        thresholds = [0.5] * CONFIG["output_dim"]
    preds = (logits >= np.array(thresholds)).astype(int)
    return preds, logits

if __name__ == "__main__":
    # text = input("Entrez le texte de l'issue : ")
    text = "This lets you use resource_stream in **enter** and **exit**, at least.\n\nFun note: turns out the python in-place config doesn't actually include the vendored pkg_resources in the PYTHONPATH. I didn't fix that in this change. It might be worth\nconsidering looking into in-place pex running instead of re-implementing pex, e.g. - in tp/py/.../pex/bin/pex.py:\n\n``` python\n    log('Running PEX file at %s with args %s' % (pex_builder.path(), args), v=options.verbosity)\n    pex = PEX(pex_builder.path(), interpreter=pex_builder.interpreter)\n    return pex.run(args=list(args))\n```\n\nSeparately, as part of (a hope!) to import the new pex, I've renamed pex.py to make_pex.py to fix the name collision.\n"
    preds, logits = predict_text(text)
    print("\nLogits :", logits)
    print("Prédictions binaires :", preds)
    predicted_labels = [target_names[i] for i, pred in enumerate(preds) if pred == 1]
    print("Labels prédits :", predicted_labels if predicted_labels else "Aucun label détecté")