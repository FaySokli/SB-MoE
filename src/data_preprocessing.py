import pandas as pd
import os
from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
import csv
import json

#Computer Science Dataset Pre-processing
corpus_path = '../multi-domain/computer_science/collection.jsonl'
corpus = pd.read_json(corpus_path, lines=True)
corpus.rename(columns={'id': '_id'}, inplace=True)
corpus['_id'] = corpus['_id'].astype(str)
corpus.to_json(corpus_path, orient='records', lines=True)

q_path = '../multi-domain/computer_science/train/queries.jsonl'
queries_train = pd.read_json(q_path, lines=True)
queries_train.rename(columns={'id': '_id'}, inplace=True)
queries_train['_id'] = queries_train['_id'].astype(str)
queries_train.to_json(q_path, orient='records', lines=True)

query_ids = queries_train['_id']
rel_doc_ids = queries_train['rel_doc_ids']
rows = []
for query_id, doc_ids in zip(query_ids, rel_doc_ids):
    for doc_id in doc_ids:
        rows.append({'query-id': query_id, 'corpus-id': doc_id, 'score': 1})

qrels_df = pd.DataFrame(rows)
qrels_df.to_csv('../multi-domain/computer_science/train/qrels.tsv', sep='\t', index=False)

q_path = '../multi-domain/computer_science/test/queries.jsonl'
queries_test = pd.read_json(q_path, lines=True)
queries_test.rename(columns={'id': '_id'}, inplace=True)
queries_test['_id'] = queries_test['_id'].astype(str)
queries_test.to_json(q_path, orient='records', lines=True)


#Political Science Dataset Pre-processing
corpus_path = '../multi-domain/political_science/collection.jsonl'
corpus = pd.read_json(corpus_path, lines=True)
corpus.rename(columns={'id': '_id'}, inplace=True)
corpus['_id'] = corpus['_id'].astype(str)
corpus.to_json(corpus_path, orient='records', lines=True)

q_path = '../multi-domain/political_science/train/queries.jsonl'
queries_train = pd.read_json(q_path, lines=True)
queries_train.rename(columns={'id': '_id'}, inplace=True)
queries_train['_id'] = queries_train['_id'].astype(str)
queries_train.to_json(q_path, orient='records', lines=True)

query_ids = queries_train['_id']
rel_doc_ids = queries_train['rel_doc_ids']
rows = []
for query_id, doc_ids in zip(query_ids, rel_doc_ids):
    for doc_id in doc_ids:
        rows.append({'query-id': query_id, 'corpus-id': doc_id, 'score': 1})

qrels_df = pd.DataFrame(rows)
qrels_df.to_csv('../multi-domain/political_science/train/qrels.tsv', sep='\t', index=False)

q_path = '../multi-domain/political_science/test/queries.jsonl'
queries_test = pd.read_json(q_path, lines=True)
queries_test.rename(columns={'id': '_id'}, inplace=True)
queries_test['_id'] = queries_test['_id'].astype(str)
queries_test.to_json(q_path, orient='records', lines=True)  

#BEIR Pre-processing
dataset = 'fever'
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
download_and_unzip(url, "../DenseIRMoE")

# QRELS
json_file = f'../DenseIRMoE/{dataset}/qrels.json'
tsv_file = f'../DenseIRMoE/{dataset}/qrels/test.tsv'

output = {}
with open(tsv_file, mode='r', newline='', encoding='utf-8') as tsv_in:
    reader = csv.DictReader(tsv_in, delimiter='\t')
    for row in reader:
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        score = int(row["score"])

        if query_id not in output:
            output[query_id] = {}
        output[query_id][corpus_id] = score

with open(json_file, mode='w', encoding='utf-8') as json_out:
    json.dump(output, json_out, indent=2)

# Test Queries
test_query_ids = set()
with open(f"../DenseIRMoE/{dataset}/qrels/test.tsv", "r", encoding="utf-8") as f:
     next(f)
     for line in f:
        query_id, _, _ = line.strip().split('\t')
        test_query_ids.add(query_id)

with open(f"../DenseIRMoE/{dataset}/queries.jsonl", "r", encoding="utf-8") as infile, \
     open(f"../DenseIRMoE/{dataset}/test_queries.jsonl", "w", encoding="utf-8") as outfile:
     
     for line in infile:
        query_obj = json.loads(line)
        if query_obj["_id"] in test_query_ids:
            json.dump(query_obj, outfile)
            outfile.write("\n")