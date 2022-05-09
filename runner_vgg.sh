#!/bin/bash

#SBATCH --job-name=vgg    # Job name
#SBATCH --output=outputs/vgg.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=16       # Schedule one core
#SBATCH --gres=gpu               # Schedule a GPU
#SBATCH --time=08:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=brown,red    # Run on either the Red or Brown queue
#SBATCH --mail-type=FAIL,END    # Send an email when the job finishes or fails

echo "No pretraining"

python src/VGG11.py --augmentation none --pretraining False

python src/VGG11.py --augmentation 01 --pretraining False

python src/VGG11.py --augmentation 02 --pretraining False

python src/VGG11.py --augmentation 03 --pretraining False

python src/VGG11.py --augmentation 04 --pretraining False

python src/VGG11.py --augmentation 05 --pretraining False

python src/VGG11.py --augmentation 06 --pretraining False

python src/VGG11.py --augmentation all --pretraining False