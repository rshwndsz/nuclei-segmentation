# U-Net

## Abstract

TODO

## Getting Started

```shell
cd dl-model-template
conda env create -f unet.yml
```

Add your data into `dataset/`.

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

## Results

TODO

## References

* [U-Net: Convolutional Networks for Biomedical Image Segmentation by Olaf Ronneberger, Philipp Fischer, Thomas Brox](https://arxiv.org/abs/1505.04597)
