# Meta-CXR
This code repository includes 3 methods used in our experiment:
The first is the Class augmentation we proposed in the paper, the CGAN used to synthesize pseudo-classes is refer to https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis. 
The second part is Reptile algorithm refer to orignal 


## Class augmentation
Class augmentation includes the training process of CGAN and the generating process of CXR pseudo-classe.  
CGAN_train.ipynb performs the 5 class Conditional GAN training and saves the model every 50 epochs.  
generate_pseudo-class.ipynb loads the Conditional GAN models and generates pseudo-classes of CXR using multiple diseases as input. 

## Reptile
Most of the settings are the same as the original setting used in Mini-imagenet. We replaced images in data\miniimagenet\train and data\miniimagenet\test with our CXR dataset. 
To run the experiment in our paper:
cd to Reptile
python -u run_miniimagenet.py --shots 50 --classes 2 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 10000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --eval-samples 50000 --checkpoint ckpt_m55

## MAML
MAML includes the MAML algorithm we used in the experiment.  
Most of the settings are the same as the original setting used in Mini-imagen
et. We replaced images in datasets\mini_imagenet_full_size\train and datasets\mini_imagenet_full_size\test with our CXR dataset.
To run the experiment in our paper:
cd to MAML/experiment_scripts
bash experiment_script.sh gpu_ids_separated_by_spaces
We use mini-imagenet_maml++-mini-imagenet_1_2_0.01_48_5_0_few_shot.sh as our setting =>
bash mini-imagenet_maml++-mini-imagenet_1_2_0.01_48_5_0_few_shot.sh 0
