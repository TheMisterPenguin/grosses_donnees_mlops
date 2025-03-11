from transformers import BertForSequenceClassification
import torch.nn as nn

class GitHubIssueClassifier(nn.Module):
    def __init__(self, num_labels, pretrained_model_name="bert-base-uncased"):
        super(GitHubIssueClassifier, self).__init__()
        # On utilise BertForSequenceClassification avec le paramètre problem_type configuré pour le multi-label
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Les logits sont retournés directement : BCEWithLogitsLoss appliquera la sigmoïde
        return outputs.logits
