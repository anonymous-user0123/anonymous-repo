

set -x

MODEL_PATH=/motion_sense/models/SFT_lora/merge_Gen_Train_SFT_wisdm_4 

python3 -m verl.trainer.main \
    config=examples/wisdm.yaml \
    data.train_files=/dataset/Train_RL_wisdm_32.json \
    data.val_files=/dataset/Train_RL_wisdm_test.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_3b_UCI_grpo \
    trainer.n_gpus_per_node=2


