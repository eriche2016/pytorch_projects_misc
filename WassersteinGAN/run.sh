# training WGAN with DCGAN
if [ 1 -eq 1 ]; then 
    python main.py --dataset cifar10 --dataroot ../datasets/cifar10/ --gpu_id 1 --cuda
fi 

# training WGAN with MLP 
if [ 1 -eq 0 ]; then 
    python main.py --mlp_G -ngf 512 
fi


