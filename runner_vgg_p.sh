echo "Pretraining"

python src/VGG11.py --augmentation none --pretraining True

python src/VGG11.py --augmentation 01 --pretraining True

python src/VGG11.py --augmentation 02 --pretraining True

python src/VGG11.py --augmentation 03 --pretraining True

python src/VGG11.py --augmentation 04 --pretraining True

python src/VGG11.py --augmentation 05 --pretraining True

python src/VGG11.py --augmentation 06 --pretraining True

python src/VGG11.py --augmentation all --pretraining True