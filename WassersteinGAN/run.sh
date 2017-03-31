# training WGAN with DCGAN
if [ 1 -eq 1 ]; then 
    python main.py --dataset cifar10 --dataroot ../datasets/cifar10/ --gpu_id 1 --cuda \
        --netG ./models_and_samples/netG_epoch_12.pth  --netD ./models_and_samples/netD_epoch_12.pth \
        --optim_state_from ./models_and_samples/optim_sate_epoch_12.pth 
fi 



# downloading lsun 
# datasets are too big, nearly 43G 
if [ 1 -eq 0 ]; then 
    python ./scripts_download/download.py --dataset lsun
fi 

if [ 1 -eq 0 ]; then 
    python main.py --dataset lsun --dataroot ../datasets/lsun/ --gpu_id 1 --cuda
fi 

# training WGAN with MLP 
if [ 1 -eq 0 ]; then 
    python main.py --mlp_G -ngf 512 
fi


