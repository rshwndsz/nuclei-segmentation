# Computational Histopathology

## Getting Started

Clone the repo
```shell
git clone https://github.com/rshwndsz/nuclei-segmentation.git nuclei-segmentation
```

Create a new conda environment and install required packages.  
```shell
cd nuclei-segmentation
conda env create -f env_<os_name>.yml
```

Download the [dataset](https://drive.google.com/drive/folders/1LSONlzWx1hMR569Zib1XwDPthnsrqfu5?usp=sharing) into `datasets/`

### Training & Validation

Train the model, with validation in between epochs, on the `train_set`.

```shell
python main.py --phase train
```

To load model from a previously saved checkpoint use: 
```shell
python main.py --phase train --load
```

### Testing

Takes a random image from the `test_set` and tests the performance of the model.

```shell
python main.py --phase test
```
