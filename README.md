## 1. Requirements

You can create an anaconda environment called BCNet with the required dependencies by running: 

  conda env create -f environment.yml
  conda activate GFLNet


## 2. Dataset

### Synthetic Dataset
- Download and visualization the Synthetic Dataset follow the instruction in  body_garment_dataset.


## 3. Training Scripts and Pre-trained Models

Training script for the `TransPIFu` baseline using the rendered DeepHuman images. 

  conda activate transpifu && cd TransPIFu/transpifu
  python -m apps.train_shape_gfl --gpu_ids 0 --name GFLNet --datasetDir ${PREFERRED_DATA_FOLDER}/data/motion_datas --save_root ${PREFERRED_DATA_FOLDER}/gfl-main/result/ --batch_size 1 --num_epoch 100 # ~ 1 day


## 6. Test Scripts

Test the models on the BUFF dataset.
