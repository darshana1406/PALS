#Benthic
python3 train_crowd.py --slice 1 --epoch 100 --num_classes 8 --batch_size 32 --lr-scheduler "step"  --noise_ratio 0.0 --partial_ratio 0.1 \
--network "R50" --lr 0.05 --wd 5e-4 --dataset "Benthic" --download True \
--noise_type "partial" --label_smoothing 0.5 --conf_th_h 0.45 --conf_th_l 0.35 \
--out ./out1 --lpi 3 \
--delta 0.75 --k_val 15 --experiment_name Benthic --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--lr-warmup-epoch 0 --start_correct 1 --lr-decay-rate 0.2 --lr-decay-epochs 60  --train_root './Benthic' --num_workers 0 --num_workers_sel 0

#Plankton
python3 train_crowd.py --slice 1 --epoch 100 --num_classes 10 --batch_size 32 --lr-scheduler "step"  --noise_ratio 0.0 --partial_ratio 0.1 \
--network "R50" --lr 0.05 --wd 5e-4 --dataset "Plankton" --download True \
--noise_type "partial" --label_smoothing 0.5 --conf_th_h 0.45 --conf_th_l 0.35 \
--out ./out1 --lpi 3 \
--delta 0.75 --k_val 15 --experiment_name Plankton --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--lr-warmup-epoch 0 --start_correct 1 --lr-decay-rate 0.2 --lr-decay-epochs 60 --train_root './Plankton' --num_workers 0 --num_workers_sel 0

#Treeversity
python3 train_crowd.py --slice 1 --epoch 100 --num_classes 6 --batch_size 32 --lr-scheduler "step"  --noise_ratio 0.0 --partial_ratio 0.1 \
--network "R50" --lr 0.05 --wd 5e-4 --dataset "Treeversity" --download True \
--noise_type "partial_v2" --label_smoothing 0.5 --conf_th_h 0.45 --conf_th_l 0.35 \
--out ./out1 --lpi 3 \
--delta 0.75 --k_val 15 --experiment_name Treeversity --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--lr-warmup-epoch 0 --start_correct 1 --lr-decay-rate 0.2 --lr-decay-epochs 60 --train_root './Treeversity#6' --num_workers 0 --num_workers_sel 0
