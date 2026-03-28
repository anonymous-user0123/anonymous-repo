

set -x

MODEL_PATH=/output/merge_Gen_UCI_SFT_train_16shot 

python3 -m verl.trainer.main \
    config=examples/config_UCI_16shot.yaml \
    data.train_files=/dataset/Gen_UCI_RL_train_16shot.json \
    data.val_files=/dataset/Gen_UCI_RL_test.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_3b_UCI_CT \
    trainer.n_gpus_per_node=2


