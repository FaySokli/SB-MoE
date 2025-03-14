import json
import logging
import os

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model.models import MoEBiEncoder
from model.utils import seed_everything

logger = logging.getLogger(__name__)


    
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)
    
    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_create_embedding_biencoder.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    
    seed_everything(cfg.general.seed)
    
    corpus = Indxr(cfg.testing.corpus_path, key_id='_id')
    corpus = sorted(corpus, key=lambda k: len(k.get("title", "") + k.get("text", "")), reverse=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    config = AutoConfig.from_pretrained(cfg.model.init.doc_model)
    config.num_experts = cfg.model.adapters.num_experts
    config.num_experts_to_use = cfg.model.adapters.num_experts_to_use
    config.adapter_residual = cfg.model.adapters.residual
    config.adapter_latent_size = cfg.model.adapters.latent_size
    config.adapter_non_linearity = cfg.model.adapters.non_linearity
    config.use_adapters = cfg.model.adapters.use_adapters
    # doc_model = MoEBertModel.from_pretrained(cfg.model.init.doc_model, config=config)
    # doc_model = BertWithMoE(cfg.model.init.doc_model, num_experts=cfg.model.init.num_experts, num_experts_to_use=cfg.model.init.num_experts_to_use)
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model, config=config)
    model = MoEBiEncoder(
        doc_model=doc_model,
        tokenizer=tokenizer,
        num_classes=cfg.model.adapters.num_experts,
        normalize=cfg.model.init.normalize,
        specialized_mode=cfg.model.init.specialized_mode,
        pooling_mode=cfg.model.init.aggregation_mode,
        use_adapters = cfg.model.adapters.use_adapters,
        device=cfg.model.init.device
    )
    if cfg.model.adapters.use_adapters:
        if cfg.model.init.specialized_mode == "variant_top1":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-variant_top1.pt', weights_only=True))
        elif cfg.model.init.specialized_mode == "variant_all":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-variant_all.pt', weights_only=True))            
        elif cfg.model.init.specialized_mode == "random":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-random.pt', weights_only=True))
    else:
        model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-ft.pt', weights_only=True))
    
    """
    logging.info(f'Loading model from {cfg.model.init.save_model}.pt')
    if os.path.exists(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt'):
        model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt'))
    else:
        logging.info('New model CLS requested, creating new checkpoint')
        torch.save(model.state_dict(), f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt')
    """
    index = 0
    texts = []
    id_to_index = {}
    # with open(cfg.testing.bm25_run_path, 'r') as f:
    #     bm25_run = json.load(f)
    
    model.eval()
    embedding_matrix = torch.zeros(len(corpus), cfg.model.init.embedding_size).float()
    for doc in tqdm.tqdm(corpus):
        
        id_to_index[doc['_id']] = index
        index += 1
        texts.append(doc.get('title','').lower() + ' ' + doc['text'].lower())
        if len(texts) == cfg.training.batch_size:
            with torch.no_grad():
                #with torch.autocast(device_type=cfg.model.init.device):
                embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()
                # embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()
            texts = []
    if texts:
        with torch.no_grad():
            # with torch.autocast(device_type=cfg.model.init.device):
            embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()
            
    
    prefix = 'fullrank'
    logging.info(f'Embedded {index} documents. Saving embedding matrix in folder {cfg.testing.embedding_dir}.')
    os.makedirs(cfg.testing.embedding_dir, exist_ok=True)
    torch.save(embedding_matrix, f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.pt')
        
    logging.info('Saving id_to_index file.')
    with open(f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.json', 'w') as f:
        json.dump(id_to_index, f)
    
if __name__ == '__main__':
    main()
    