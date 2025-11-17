DATASET=computer_science
MODEL=tinybert

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_all
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_all
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_all


DATASET=political_science
MODEL=contriever-base

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_top1
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_top1
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_top1


DATASET=hotpotqa
MODEL=bert-base

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_all
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_all
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_all


DATASET=nq-train
MODEL=tinybert

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_top1
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_top1
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6 model.init.specialized_mode=sbmoe_top1

