#CIFAR-100
python3 train.py --epoch 500 --num_classes 100 --batch_size 256 --lr-scheduler "cosine"  --noise_ratio 0.3 --partial_ratio 0.05 \
--network "R18" --lr 0.1 --wd 1e-3 --dataset "CIFAR-100" --download True \
--noise_type "partial" --label_smoothing 0.5 --conf_th_h 0.45 --conf_th_l 0.35  --out ./out \
--delta 0.25 --k_val 15 --experiment_name CIFAR100 --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--lr-warmup-epoch 0 --start_correct 1

#CIFAR-10
python3 train.py --epoch 500 --num_classes 10 --batch_size 256 --lr-scheduler "cosine"  --noise_ratio 0.3 --partial_ratio 0.5 \
--network "R18" --lr 0.1 --wd 1e-3 --dataset "CIFAR-10" --download True \
--noise_type "partial" --label_smoothing 0.5 --conf_th_h 0.45 --conf_th_l 0.35  --out ./out \
--delta 0.25 --k_val 15 --experiment_name CIFAR10 --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--lr-warmup-epoch 0 --start_correct 1

#CIFAR-100h
python3 train.py --epoch 500 --num_classes 100 --batch_size 256 --lr-scheduler "cosine"  --noise_ratio 0.2 --partial_ratio 0.5 --heirarchical True \
--network "R18" --lr 0.1 --wd 1e-3 --dataset "CIFAR-100" --download True \
--noise_type "partial" --label_smoothing 0.5 --conf_th_h 0.45 --conf_th_l 0.35 --out ./out \
--delta 0.25 --k_val 15 --experiment_name CIFAR100h --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--lr-warmup-epoch 0 --start_correct 1

#CUB-200
python3 train.py --epoch 250 --num_classes 200 --batch_size 64 --lr-scheduler "step"  --noise_ratio 0.2 --partial_ratio 0.05 --num_workers 0 --num_workers_sel 0 \
--network "R18" --lr 0.05 --wd 5e-4 --dataset "CUB-200" --download True \
--noise_type "partial" --label_smoothing 0.5 --conf_th_h 0.45 --conf_th_l 0.35   --out ./out \
--delta 0.25 --k_val 15 --experiment_name CUB200 --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--lr-warmup-epoch 0 --start_correct 1 --lr-decay-rate 0.2 --lr-decay-epochs 60 120 160 200

