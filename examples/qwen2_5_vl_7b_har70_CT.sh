




MODEL_PATH=/har70/models/SFT_lora/har70_Qwen25_SFT_16_merge

python3 -m verl.trainer.main \
    config=examples/config_har70.yaml \
    data.train_files=/dataset/Train_Gen_RL_har70_16.json \
    data.val_files=/dataset/Train_Gen_RL_har70_test.json \
    data.mini_rollout_batch_size=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    trainer.experiment_name=qwen2_5_vl_3b_har70 \
    trainer.n_gpus_per_node=2
