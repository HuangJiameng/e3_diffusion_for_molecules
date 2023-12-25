SHELL := /bin/bash

calc_steps_original:
	${eval EXP_NAME:= without_equi_eot}
	$(eval GPU_ID := 2)
	python  print.py --n_epochs 3000 --exp_name fm_qm9  --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 100 --nf 256 --n_layers 9 --lr 1e-4 --dequantization deterministic  --normalize_factors "[1,4,10]" --test_epochs 200 --ema_decay 0.9999 --probabilistic_model flow_matching --visualize_every_batch 1000 --exp_name ${EXP_NAME} --discrete_path HB_path  --dp "False"  --gpu_id ${GPU_ID}  --sample_eva_epochs 20  --output_dir ./outputs --resume  ./checkpoints --start_epoch 0 --ode_method dopri5 --node_classifier_model_ckpt "" --n_stability_samples 1000 --no_wandb




