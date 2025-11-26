import json
import logging
import os
import ipdb

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model.models import MoEBiEncoder
from model.utils import seed_everything

from ranx import Run, Qrels, compare

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

def visualize_tsne(query_embedding, top_doc_embeddings, top_doc_ids, query_id, output_dir, experts_used, relevants, use_adapters=True):
    import matplotlib.lines as mlines

    all_embeddings = torch.cat([query_embedding, top_doc_embeddings], dim=0).cpu().numpy()
    relevant_set = set(relevants)

    tsne = TSNE(n_components=3, random_state=42, perplexity=30, init='pca')
    embeddings_3d = tsne.fit_transform(all_embeddings)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(embeddings_3d[0, 0], embeddings_3d[0, 1], embeddings_3d[0, 2], 
               c='black', label='query', marker='X', s=100)

    color_map = plt.cm.tab10
    marker_list = ['o', '^', 's', 'D', 'v', 'P', '*', 'X', '<', '>']

    for i, (x, y, z) in enumerate(embeddings_3d[1:], start=1):
        doc_id = top_doc_ids[i-1]
        is_relevant = doc_id in relevant_set

        if use_adapters:
            expert_id = experts_used[i-1]
            color = color_map(expert_id)
            marker = marker_list[expert_id % len(marker_list)]
        else:
            color = 'blue'
            marker = 'o'

        if is_relevant:
            ax.scatter(x, y, z, edgecolors='red', facecolors='none', s=120, linewidths=2,
                       marker='o', label='relevant_docs' if i == 1 else "")
        else:
            ax.scatter(x, y, z, c=[color], marker=marker, alpha=0.6, s=40)

    # Legend
    handles = [mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='query'),
               mlines.Line2D([], [], color='red', marker='o', markerfacecolor='none', linestyle='None', markersize=10, label='relevant_docs')]

    if use_adapters:
        unique_experts = sorted(set(experts_used))
        for expert_id in unique_experts:
            handles.append(
                mlines.Line2D([], [], color=color_map(expert_id), marker=marker_list[expert_id % len(marker_list)],
                              linestyle='None', markersize=10, label=f'Expert {expert_id + 1}')
            )

    ax.legend(handles=handles, fontsize=20)
    # ax.set_title(f"3D t-SNE: Query {query_id} and Top 1000 Docs")
    plt.grid(True)
    plt.tight_layout()

    filename = f"tsne_query_{query_id}_3D_experts.png" if use_adapters else f"tsne_query_{query_id}_3D_NO_experts.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=900)
    plt.close()
    print(f"Saved t-SNE plot: {save_path}")


def get_bert_rerank(data, model, doc_embedding, bm25_runs, id_to_index):
    bert_run = {}
    model.eval()
    for d in tqdm.tqdm(data, total=len(data)):
        with torch.no_grad():
            q_embedding = model.query_encoder([d['text']])
            
        bm25_docs = list(bm25_runs[d['_id']].keys())
        d_embeddings = doc_embedding[torch.tensor([int(id_to_index[x]) for x in bm25_docs])]
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
        bert_run[d['_id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs)}
        
    return bert_run


def get_full_bert_rank(data, model, doc_embedding, id_to_index, k=1000):
    bert_run = {}
    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    model.eval()
    for d in tqdm.tqdm(data, total=len(data)):
        with torch.no_grad():
            # with torch.autocast(device_type=model.device):
            q_embedding = model.query_encoder([d['text']])
        
        bert_scores = torch.einsum('xy, ly -> x', doc_embedding, q_embedding)
        index_sorted = torch.argsort(bert_scores, descending=True)
        top_k = index_sorted[:k]
        bert_ids = [index_to_id[int(_id)] for _id in top_k]
        bert_scores = bert_scores[top_k]
        bert_run[d['_id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bert_ids)}
        
        
    return bert_run
    
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)
    
    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_testing_biencoder.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    config = AutoConfig.from_pretrained(cfg.model.init.doc_model)
    config.num_experts = cfg.model.adapters.num_experts
    config.adapter_residual = cfg.model.adapters.residual
    config.adapter_latent_size = cfg.model.adapters.latent_size
    config.adapter_non_linearity = cfg.model.adapters.non_linearity
    config.use_adapters = cfg.model.adapters.use_adapters
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model)
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
        if cfg.model.init.specialized_mode == "sbmoe_top1":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-sbmoe_top1.pt', weights_only=True))
            print(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-sbmoe_top1.pt')
        elif cfg.model.init.specialized_mode == "sbmoe_all":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-sbmoe_top1.pt', weights_only=True))
            print(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-sbmoe_top1.pt')
        elif cfg.model.init.specialized_mode == "random":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-random.pt', weights_only=True))
            print(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-random.pt')
    else:
        model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-ft.pt', weights_only=True))
        print(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-ft.pt')
    

    doc_embedding = torch.load(f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_fullrank.pt', weights_only=True).to(cfg.model.init.device)
    
    with open(f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_fullrank.json', 'r') as f:
        id_to_index = json.load(f)
    
    # with open(cfg.testing.bm25_run_path, 'r') as f:
    #     bm25_run = json.load(f)
    
    data = Indxr(cfg.testing.query_path, key_id='_id')
    if cfg.testing.rerank:
        bert_run = get_bert_rerank(data, model, doc_embedding, bm25_run, id_to_index)
    else:
        bert_run = get_full_bert_rank(data, model, doc_embedding, id_to_index, 1000)
        
    
    # with open(f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}_biencoder.json', 'w') as f:
    #     json.dump(bert_run, f)
        
        
    ranx_qrels = Qrels.from_file(cfg.testing.qrels_path)
    
    if cfg.testing.rerank:
        ranx_run = Run(bert_run, 'ReRanker')
        ranx_bm25_run = Run(bm25_run, name='BM25')
        models = [ranx_bm25_run, ranx_run]
    else:
        ranx_run = Run(bert_run, 'FullRun')
        models = [ranx_run]

    if cfg.model.adapters.use_adapters:
        ranx_run.save(f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_biencoder-{cfg.model.init.specialized_mode}.json')
    else:
        ranx_run.save(f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_biencoder-ft.json')
    
    evaluation_report = compare(ranx_qrels, models, ['map@100', 'mrr@10', 'recall@100', 'ndcg@10', 'precision@1', 'ndcg@3'])
    print(evaluation_report)
    logging.info(f"Results for {cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_biencoder.json:\n{evaluation_report}")

    ############################
    # Create directory for t-SNE plots
    tsne_dir = os.path.join(cfg.dataset.output_dir, 'tsne_plots')
    os.makedirs(tsne_dir, exist_ok=True)

    # Load .npy t-SNE files
    tsne_data_dir = cfg.testing.embedding_dir
    prefix = "fullrank"  # adjust if needed (e.g., rerank)

    # Load embeddings and expert_ids
    embedding_tsne_path = os.path.join(tsne_data_dir, f"doc_embeddings_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.npy")
    expert_ids_path = os.path.join(tsne_data_dir, f"expert_ids_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.npy")

    all_doc_embeddings_np = np.load(embedding_tsne_path)
    all_expert_ids = np.load(expert_ids_path)
    all_doc_embeddings = torch.from_numpy(all_doc_embeddings_np).to(cfg.model.init.device)

    # for i in range(30):
    import time
    random.seed(time.time())  # Changes every run
    random_query = random.choice(data)
    query_id = random_query['_id']
    # query_id = "2950442623"
    # random_query = data.get(query_id)
    print(f"Selected query ID: {query_id}")

    # query
    model.eval()
    with torch.no_grad():
        query_embedding = model.query_encoder([random_query['text']]).to(cfg.model.init.device)

    # Get top 1000 docs
    topk_ids = list(bert_run[query_id].keys())[:1000]
    topk_indices = [id_to_index[doc_id] for doc_id in topk_ids]
    top_doc_embeddings = doc_embedding[topk_indices]
    top_doc_expert_ids = [int(all_expert_ids[i]) for i in topk_indices]
    relevants_dict = ranx_qrels.to_dict()
    relevants = set(relevants_dict.get(query_id, {}).keys())

    # Generate and save t-SNE
    visualize_tsne(query_embedding, top_doc_embeddings, topk_ids, query_id, tsne_dir, top_doc_expert_ids, relevants, use_adapters=cfg.model.adapters.use_adapters)
    ############################

if __name__ == '__main__':
    main()
