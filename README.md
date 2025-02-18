# Mixture of Experts Approaches in Dense Retrieval Tasks
To reproduce the experiments complete the following steps:

## Step 1:
Run the following command to complete data pre-processing and bring all datasets to the BEIR format: <br>
```python3 data_preprocessing.py``` <br><br>
For our experiments we use the following four publically available IR benchmarks: (i) HotpotQA and (ii) Natural Questions from the [BEIR collection](https://arxiv.org/abs/2104.08663), as well as the (iii) Political Science and (iv) Computer Science collections from the [Multi-Domain Benchmark](https://dl.acm.org/doi/abs/10.1145/3511808.3557536). We bring the last two collections to the BEIR format for our experiments.

## Step 2:
Update the parameters in the ```pipeline.sh``` file to the desired ones and execute the script. <br><br>
For example, the following command will train *variant_ALL* on TinyBERT using the HotpotQA dataset by employing 6 experts: <br>
```python3 1_train_new_moe.py model=tinybert dataset=hotpotqa testing=hotpotqa model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=variant_all``` <br><br>
_*Note: When use_adapters=True, only two modes are allowed; (i) 'variant_all' and (ii) 'variant_top1'.