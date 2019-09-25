# U-Net

## Getting Started

Clone the repo
```shell
git clone https://github.com/rshwndsz/nuclei-segmentation.git nuclei-segmentation
```

Create a new conda environment and install required packages.  
```shell
cd nuclei-segmentation
conda env create -f unet_<os_name>.yml
```

Download the dataset from https://drive.google.com/drive/folders/1LSONlzWx1hMR569Zib1XwDPthnsrqfu5?usp=sharing  
Add it into `dataset/`.

### Training

```shell
python main.py --phase train
```

### Validation

```shell
python main.py --phase val
```

### Testing

```shell
python main.py --phase test --in_path xxx.jpg  --out_path results
```
