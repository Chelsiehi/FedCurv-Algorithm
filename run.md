
# 实验设置

## mnist iid
B = 10 E = 1/5/20
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 1 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 1 --iid --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 5 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 5 --iid --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 20 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 20 --iid --alg fedcurv
B = 50 E = 1/5/20
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 1 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 1 --iid --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 5 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 5 --iid --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 20 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 20 --iid --alg fedcurv

## mnist non-iid
B = 10 E = 1/5/20
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 1 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 1 --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 5 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 5 --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 20 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 20 --alg fedcurv

B = 50 E = 1/5/20
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 1 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 1 --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 5 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 5 --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 20 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 20 --alg fedcurv


## cifar10 iid
B = 10 E = 1/5/20
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 1 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 1 --iid --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 5 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 5 --iid --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 20 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 20 --iid --alg fedcurv

B = 50 E = 1/5/20
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 1 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 1 --iid --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 5 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 5 --iid --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 20 --iid --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 20 --iid --alg fedcurv

## cifar10 non-iid
B = 10 E = 1/5/2
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 1 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 1 --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 5 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 5 --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 20 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 20 --alg fedcurv

B = 50 E = 1/5/20
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 1 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 1 --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 5 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 5 --alg fedcurv
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 20 --alg fedavg
python3 main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 20 --alg fedcurv
