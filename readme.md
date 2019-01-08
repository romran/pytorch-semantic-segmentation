# Semantic Segmentation using [Digital Retinal Images](http://www.isi.uu.nl/Research/Databases/DRIVE/) 

## Model

Results could be compared between two networks:
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [FusionNet: A deep fully residual convolutional neural network for image segmentation in connectomics](https://arxiv.org/abs/1612.05360)

## Download data

Download DRIVE (Digital Retinal Images for Vessel Extraction) dataset from http://www.isi.uu.nl/Research/Databases/DRIVE/

or you could use your own dataset for training, take a look at [notebook](https://github.com/romran/pytorch-semantic-segmentation/blob/master/prepare_data.ipynb) to prepare images.

## Quick setup (for Windows)

- With anaconda create virtual environment `conda create -n myenv python=3.5` and `activate myenv`
- Install PyTorch `conda install pytorch torchvision -c pytorch `
- Install all remaining packages with `pip install -r requirements.txt`

## Make required directory

- `mkdir result` to save predictions
- `mkdir model` to save trained model

## Arguments

- `--network, default="fusionnet", help="choose between fusionnet or unet"`
- `--batch_size, default=1, help="batch size"`
- `--num_gpu, default=1, help="number of gpus"`

## Train Model

`python main.py --network unet --batch_size 1 --num_gpu 1`
 
## References 

- The implementation is heavily influenced by [Kind-PyTorch-Tutorial](https://github.com/GunhoChoi/Kind-PyTorch-Tutorial) 
- [J.J. Staal, M.D. Abramoff, M. Niemeijer, M.A. Viergever, B. van Ginneken, "Ridge based vessel segmentation in color images of the retina", IEEE Transactions on Medical Imaging, 2004, vol. 23, pp. 501-509.](http://www.isi.uu.nl/Research/Databases/DRIVE/id=855.html)
- [DRIVE dataset](http://www.isi.uu.nl/Research/Databases/DRIVE/) 