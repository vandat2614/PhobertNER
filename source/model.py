import torch
import torch.nn as nn
from transformers import AutoModel

class NerModel(nn.Module):
    def __init__(self, config, num_labels):
        super(NerModel, self).__init__()
        
        # Load backbone from config
        model_name = config['MODEL']['PRETRAINED_NAME']
        self.phobert = AutoModel.from_pretrained(model_name)
        
        bert_output_dim = config['MODEL']['HIDDEN_DIM']
        dropout_prob = config['MODEL']['DROPOUT']

        # Freeze pretrained model
        if config['MODEL']['FREEZE_BERT']:
            for param in self.phobert.parameters():
                param.requires_grad = False
            
            n_unfreeze = config['MODEL']['UNFREEZE_LAST_N_LAYERS']
            if n_unfreeze > 0:
                layers_to_unfreeze = [f'encoder.layer.{11 - i}' for i in range(n_unfreeze)]
                
                for name, param in self.phobert.named_parameters():
                    if any(layer_name in name for layer_name in layers_to_unfreeze):
                        param.requires_grad = True

        self.dropout = nn.Dropout(dropout_prob)
        
        # Build classifier head
        num_layers = config['MODEL']['CLASSIFIER']['NUM_LAYERS']
        if num_layers == 1:
            self.classifier = nn.Linear(bert_output_dim, num_labels)

        else:
            hidden_dim = config['MODEL']['CLASSIFIER']['HIDDEN_DIM']

            layers = []
            layers.append(nn.Linear(bert_output_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_prob))

            layers.append(nn.Linear(hidden_dim, num_labels))
            self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):

        outputs = self.phobert( 
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # output shape: [batch_size, seq_len, hidden_size]
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # logits shape: [batch_size, seq_len, num_labels]
        logits = self.classifier(sequence_output)
        
        return logits