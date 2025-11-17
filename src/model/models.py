import torch
from torch import clamp as t_clamp
from torch import nn as nn
from torch import sum as t_sum
from torch import max as t_max
from torch import einsum
from torch.nn import functional as F

class Specializer(nn.Module):
    def __init__(self, hidden_size, device):
        super(Specializer, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        self.query_embedding_changer_1 = nn.Linear(self.hidden_size, self.hidden_size//2).to(device)
        self.query_embedding_changer_4 = nn.Linear(self.hidden_size//2, self.hidden_size).to(device)
        
    def forward(self, query_embs):
        query_embs_1 = F.relu(self.query_embedding_changer_1(query_embs))
        query_embs_2 = self.query_embedding_changer_4(query_embs_1)
        
        return query_embs_2


class MoEBiEncoder(nn.Module):
    def __init__(
        self,
        doc_model,
        tokenizer,
        num_classes,
        max_tokens=512,
        normalize=False,
        specialized_mode='sbmoe_top1',
        pooling_mode='mean',
        use_adapters=True,
        track_expert_usage=False,
        device='cpu',
    ):
        super(MoEBiEncoder, self).__init__()
        self.doc_model = doc_model.to(device)
        self.hidden_size = self.doc_model.config.hidden_size
        self.tokenizer = tokenizer
        self.device = device
        self.normalize = normalize
        self.max_tokens = max_tokens
        self.use_adapters = use_adapters
        assert specialized_mode in ['sbmoe_top1', 'sbmoe_all', 'random'], 'Only random, sbmoe_top1 and sbmoe_all specialzed modes allowed'
        self.specialized_mode = specialized_mode
        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        if pooling_mode == 'mean':
            self.pooling = self.mean_pooling
        elif pooling_mode == 'max':
            self.pooling = self.max_pooling
        elif pooling_mode == 'cls':
            self.pooling = self.cls_pooling
        elif pooling_mode == 'identity':
            self.pooling = self.identity
        
        self.num_classes = num_classes
        self.expert_usage_counter = torch.zeros(self.num_classes).to(self.device)
        self.track_expert_usage = track_expert_usage
        self.init_cls()
        
        self.specializer = nn.ModuleList([Specializer(self.hidden_size, self.device) for _ in range(self.num_classes)])    
        
        
    def query_encoder_no_moe(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])
        
    
    def init_cls(self):
        self.cls_1 = nn.Linear(self.hidden_size, self.hidden_size//2).to(self.device)
        # self.cls_2 = nn.Linear(self.hidden_size*2, self.hidden_size*4).to(self.device)
        self.cls_3 = nn.Linear(self.hidden_size//2, self.num_classes).to(self.device)
        self.noise_linear = nn.Linear(self.hidden_size, self.num_classes).to(self.device)
        
    
    def query_encoder(self, sentences):
        query_embedding = self.query_encoder_no_moe(sentences)
        if self.use_adapters:
            query_class = self.cls(query_embedding)
            if self.specialized_mode == "random":
                query_class = torch.rand(query_embedding.shape[0], self.num_classes).to(self.device)
                # TOP-k GATING
                topk_values, topk_indices = torch.topk(query_class, 1, dim=1)
                mask = torch.zeros_like(query_class).scatter_(1, topk_indices, 1)
                
                # Multiply the original output with the mask to keep only the max value
                query_class = query_class * mask
            query_embedding = self.query_embedder(query_embedding, query_class)
        return query_embedding
        

    def doc_encoder_no_moe(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])

    def doc_encoder(self, sentences):
        doc_embedding = self.doc_encoder_no_moe(sentences)
        if self.use_adapters:
            doc_class = self.cls(doc_embedding)
            if self.specialized_mode == "random":
                doc_class = torch.rand(doc_embedding.shape[0], self.num_classes).to(self.device)
                # TOP-k GATING
                topk_values, topk_indices = torch.topk(doc_class, 1, dim=1)
                mask = torch.zeros_like(doc_class).scatter_(1, topk_indices, 1)
                
                # Multiply the original output with the mask to keep only the max value
                doc_class = doc_class * mask
            doc_embedding = self.doc_embedder(doc_embedding, doc_class)
        return doc_embedding
        

    def cls(self, query_embedding):
        x1 = F.relu(self.cls_1(query_embedding))
        # x2 = F.relu(self.cls_2(x1))
        out = self.cls_3(x1)

        if self.training:
            noise_logits = self.noise_linear(query_embedding)
            noise = torch.randn_like(out)*F.softplus(noise_logits)
            noisy_logits = out + noise

            noisy_logits = torch.softmax(noisy_logits, dim=-1)

            # TOP-k GATING
            topk_values, topk_indices = torch.topk(noisy_logits, 1, dim=1)
            mask = torch.zeros_like(noisy_logits).scatter_(1, topk_indices, 1)
            
            noisy_logits = noisy_logits * mask
            return noisy_logits
        
        else:
            if self.specialized_mode == 'sbmoe_top1':
                out = torch.softmax(out, dim=-1)

                # TOP-k GATING
                topk_values, topk_indices = torch.topk(out, 1, dim=1)
                mask = torch.zeros_like(out).scatter_(1, topk_indices, 1)
                
                out = out * mask
                return out
            
            elif self.specialized_mode == 'sbmoe_all':
                out = torch.softmax(out, dim=-1)
                if self.track_expert_usage:
                    top_expert = out.argmax(dim=1)
                    for idx in top_expert:
                        self.expert_usage_counter[idx] += 1
                return out
    

    def forward(self, data):
        query_embedding = self.query_encoder(data[0])
        pos_embedding = self.doc_encoder(data[1])

        return query_embedding, pos_embedding

    def val_forward(self, data):
        query_embedding = self.query_encoder(data[0])
        pos_embedding = self.doc_encoder(data[1])

        return query_embedding, pos_embedding


    def query_embedder(self, query_embedding, query_class):
        query_embs = [self.specializer[i](query_embedding) for i in range(self.num_classes)]
        
        query_embs = torch.stack(query_embs, dim=1)
        
        query_class = query_class #sigmoid(query_class) # softmax(query_class, dim=-1)

        query_embs = F.normalize(einsum('bmd,bm->bd', query_embs, query_class), dim=-1, eps=1e-6) + query_embedding

        if self.normalize:
            return F.normalize(query_embs, dim=-1)
        return query_embs
    
    def doc_embedder(self, doc_embedding, doc_class):
        return self.query_embedder(doc_embedding, doc_class)
        

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)


    @staticmethod
    def cls_pooling(model_output, attention_mask):
        last_hidden = model_output["last_hidden_state"]
        # last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden[:, 0]


    @staticmethod
    def identity(model_output, attention_mask):
        return model_output['pooler_output']
    

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]
