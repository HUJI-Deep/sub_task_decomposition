#!/bin/bash

if test $num_of_bits -eq 8
then
    logging_freq_flag="--check_val_every_n_epoch 1"
    eval_steps=1
elif test $num_of_bits -eq 10
then
    logging_freq_flag="--val_check_interval 10"
    eval_steps=4
elif test $num_of_bits -eq 12
then
    logging_freq_flag="--val_check_interval 10"
    eval_steps=16
elif test $num_of_bits -eq 14
then
    logging_freq_flag="--check_val_every_n_epoch 1"
    eval_steps=32
elif test $num_of_bits -eq 16
then
    if $steps
    then
	logging_freq_flag="--val_check_interval 10"
    else
	logging_freq_flag="--val_check_interval 1000"
    fi
    eval_steps=32
elif test $num_of_bits -eq 32
then
    if $steps
    then
        logging_freq_flag="--val_check_interval 10"
    else
        logging_freq_flag="--val_check_interval 1000"
    fi
    eval_steps=32
else
    logging_freq_flag="--val_check_interval 1000"
    eval_steps=32
fi

if test $num_of_bits -eq 256
then
    train_batch_size=16
    accumulate_grad_batches=2
elif test $num_of_bits -eq 256
then
	train_batch_size=8
	accumulate_grad_batches=4
else
    train_batch_size=32
    accumulate_grad_batches=1
fi

if $steps
then
        steps_str="step_by_step"
else
        steps_str="single_step"
fi

if $greedy_decoding
then
	decoding_algorithm_flag="--evaluate_with_greedy_decoding"
else
	decoding_algorithm_flag="--evaluate_with_sampling"
fi

python bit_subset_parity_trainer.py --${steps_str} --gpus 1 --default_root_dir bit_subset_parity_results/${steps_str}/${num_of_bits}_bits/seed_${seed}_steps_${training_iterations}_lr_${learning_rate}_wd_${weight_decay}_depth_${depth}_width_${width}_num_heads_${num_heads}_warmup_steps_${warmup_steps}_greedy_decoding_${greedy_decoding} --benchmark true --max_steps $training_iterations --min_steps $training_iterations --max_epochs $training_iterations  --gradient_clip_val 1.  --learning_rate $learning_rate --num_of_bits $num_of_bits --weight_decay $weight_decay --depth $depth --width $width --num_heads $num_heads --seed $seed --eval_steps $eval_steps $logging_freq_flag --train_batch_size $train_batch_size --accumulate_grad_batches $accumulate_grad_batches --warmup_steps $warmup_steps $decoding_algorithm_flag $additional_args

