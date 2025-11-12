# Mixture of Experts Approaches in Dense Retrieval Tasks
Read the paper [here](https://arxiv.org/abs/2510.15683)
To reproduce the experiments, complete the following steps:

## Step 1:
Run the following command to complete data pre-processing and bring all datasets to the BEIR format: <br>
```python3 data_preprocessing.py``` <br><br>
For our experiments, we use the following four publicly available IR benchmarks: (i) HotpotQA and (ii) Natural Questions from the [BEIR collection](https://arxiv.org/abs/2104.08663), as well as the (iii) Political Science and (iv) Computer Science collections from the [Multi-Domain Benchmark](https://dl.acm.org/doi/abs/10.1145/3511808.3557536). We bring the last two collections to the BEIR format for our experiments.

## Step 2:
Update the parameters in the ```pipeline.sh``` file to the desired ones and execute the script. <br><br>
For example, the following command will train *variant_ALL* on TinyBERT using the HotpotQA dataset by employing 6 experts: <br>
```python3 1_train_new_moe.py model=tinybert dataset=hotpotqa testing=hotpotqa model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=variant_all``` <br><br>
_*Note: When use_adapters=True, only two modes are allowed; (i) 'variant_all' and (ii) 'variant_top1'.

## 3-Dimensional T-SNE plots
3D t-SNE representations of the same query and its top 1000 documents, on the left embedded by the original DRM
and on the right by our model, for all seven benchmarks.
### MS MARCO
<p float="left">
  <img src="tsne/msmarco/_00tsne_query_541425_3D_NO_experts.png" width="45%" />
  <img src="tsne/msmarco/_00tsne_query_541425_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/msmarco/_02tsne_query_638795_3D_NO_experts.png" width="45%" />
  <img src="tsne/msmarco/_02tsne_query_638795_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/msmarco/_05tsne_query_946428_3D_NO_experts.png" width="45%" />
  <img src="tsne/msmarco/_05tsne_query_946428_3D_experts.png" width="45%" />
</p>

### TREC DL 2019
<p float="left">
  <img src="tsne/trec19/_00trec19tsne_query_183378_3D_NO_experts.png" width="45%" />
  <img src="tsne/trec19/_00trec19tsne_query_183378_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/trec19/_06trec19tsne_query_490595_3D_NO_experts.png" width="45%" />
  <img src="tsne/trec19/_06trec19tsne_query_490595_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/trec19/_07trec19tsne_query_527433_3D_NO_experts.png" width="45%" />
  <img src="tsne/trec19/_07trec19tsne_query_527433_3D_experts.png" width="45%" />
</p>

### TREC DL 2020
<p float="left">
  <img src="tsne/trec20/_01trec20tsne_query_135802_3D_NO_experts.png" width="45%" />
  <img src="tsne/trec20/_01trec20tsne_query_135802_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/trec20/_03trec20tsne_query_1116380_3D_NO_experts.png" width="45%" />
  <img src="tsne/trec20/_03trec20tsne_query_1116380_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/trec20/_07trec20tsne_query_1133579_3D_NO_experts.png" width="45%" />
  <img src="tsne/trec20/_07trec20tsne_query_1133579_3D_experts.png" width="45%" />
</p>

### Natural Questions
<p float="left">
  <img src="tsne/nq/_02tsne_query_test389_3D_NO_experts.png" width="45%" />
  <img src="tsne/nq/_02tsne_query_test389_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/nq/_07tsne_query_test3012_3D_NO_experts.png" width="45%" />
  <img src="tsne/nq/_07tsne_query_test3012_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/nq/_09tsne_query_test1662_3D_NO_experts.png" width="45%" />
  <img src="tsne/nq/_09tsne_query_test1662_3D_experts.png" width="45%" />
</p>

### HotpotQA
<p float="left">
  <img src="tsne/hotpotqa/00tsne_query_5ab1c693554299722f9b4c66_3D_NO_experts.png" width="45%" />
  <img src="tsne/hotpotqa/00tsne_query_5ab1c693554299722f9b4c66_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/hotpotqa/01tsne_query_5a7b116a554299042af8f6c4_3D_NO_experts.png" width="45%" />
  <img src="tsne/hotpotqa/01tsne_query_5a7b116a554299042af8f6c4_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/hotpotqa/05tsne_query_5ab4147a5542996a3a969f1e_3D_NO_experts.png" width="45%" />
  <img src="tsne/hotpotqa/05tsne_query_5ab4147a5542996a3a969f1e_3D_experts.png" width="45%" />
</p>

### Political Science
<p float="left">
  <img src="tsne/ps/00tsne_query_2620554973_3D_NO_experts.png" width="45%" />
  <img src="tsne/ps/00tsne_query_2620554973_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/ps/02tsne_query_2394742722_3D_NO_experts.png" width="45%" />
  <img src="tsne/ps/02tsne_query_2394742722_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/ps/03tsne_query_2617258900_3D_NO_experts.png" width="45%" />
  <img src="tsne/ps/03tsne_query_2617258900_3D_experts.png" width="45%" />
</p>

### Computer Science
<p float="left">
  <img src="tsne/cs/_00tsne_query_2606965392_3D_NO_experts.png" width="45%" />
  <img src="tsne/cs/_00tsne_query_2606965392_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/cs/_03tsne_query_2761208857_3D_NO_experts.png" width="45%" />
  <img src="tsne/cs/_03tsne_query_2761208857_3D_experts.png" width="45%" />
</p>
<p float="left">
  <img src="tsne/cs/_07tsne_query_2539319785_3D_NO_experts.png" width="45%" />
  <img src="tsne/cs/_07tsne_query_2539319785_3D_experts.png" width="45%" />
</p>
