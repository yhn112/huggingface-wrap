import torch
from transformers.optimization import AdamW

BERT_KEYS = ["input_ids", "token_type_ids", "attention_mask"]

class BertClassifier(torch.nn.Module):
    
    def __init__(self, model, state_key="last_hidden_state", lr=1e-5, clip=1.0, accumulate_gradients=1):
        super(BertClassifier, self).__init__()
        self.state_key = state_key
        self.model = model
        self.proj_layer = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        # self.criterion = torch.nn.BCELoss(reduction="sum")
        self.criterion = torch.nn.BCELoss(reduction="mean")
        self.optimizer = AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.clip = clip
        self.accumulate_gradients = accumulate_gradients
        self._batches_accumulated = 0
        
    @property
    def hidden_size(self):
        return self.model.config.hidden_size
    
    def forward(self, batch):
        batch = {key: value for key, value in batch.items() if key in BERT_KEYS}
        bert_outputs = self.model(**batch)
        logit = self.proj_layer(bert_outputs[self.state_key])
        if logit.dim() == 3:
            logit = logit[:,0]
        prob = self.sigmoid(logit)[:,0]
        return prob
    
    def train_on_batch(self, batch):
        self.model.train()
        if self._batches_accumulated == 0:
            self.optimizer.zero_grad()
        outputs = self._validate(batch)
        outputs["loss"] /= self.accumulate_gradients
        outputs["loss"].backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self._batches_accumulated = (self._batches_accumulated + 1) % self.accumulate_gradients
        if self._batches_accumulated == 0:
            self.optimizer.step()
            # self.optimizer.zero_grad()
        return outputs
    
    def validate_on_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self._validate(batch)
        
    def _validate(self, batch):
        probs = self.forward(batch)
        loss = self.criterion(probs, batch["labels"])
        return {"probs": probs, "loss": loss, "loss_value": loss.item()}

    def predict_on_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            probs = self.forward(batch).cpu().numpy()
        labels = (probs > 0.5).astype(int)
        return {"probs": probs, "labels": labels}