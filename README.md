# Evaluating Same-Different Models
This simple codebase allows you to load a fine-tuned model from the paper [Deep Neural Networks Can Learn Generalizable Same-Different Visual Relations](https://arxiv.org/abs/2310.09612) by Tartaglini et al. (2023). 

## Getting started
Begin by cloning this repo onto your local machine. Navigate to the directory where you would like this codebase to be installed and run: 
```
git clone https://github.com/alexatartaglini/same-different-eval.git
```
Next, make sure that you've installed all required Python packages. Create a new virtual environment, activate the environment, and run:
```
pip install -r requirements.txt
```

## Adding an evaluation dataset
Evaluation datasets are stored in the directory `datasets`. This repo comes with a default same-different evaluation dataset named `filled`, which demonstrates how evaluation datasets should be formatted. This dataset contains images like the following:

<div align="center">
  
| "Same": | ![Same 1](./datasets/filled/same/0HXEDG3A.png) | ![Same 2](./datasets/filled/same/2N5QSAVE.png) | ![Same 3](./datasets/filled/same/3GOM1QKQ.png) |
|-|-|-|-|
| **"Different":** | ![Diff 1](./datasets/filled/different/0PA4A9BN.png) | ![Diff 2](./datasets/filled/different/1VQRONT8.png) | ![Diff 3](./datasets/filled/different/2Y2QZYII.png) |

</div>

In order to add your own evaluation dataset, create a subdirectory inside `datasets` with the name of your dataset (e.g. `my_dataset`). Then, inside `my_dataset`, create two more subdirectories named `same` and `different`. Store all of your dataset's "same" images inside `same` and all of the "different" images inside `different`. 

**Important: images should be .png files.** 
*(If you need to use other image formats, change the command on line 73 of `evaluate.py`.)*

## Models
This repo comes with the model weights for all median seeds reported in [Tartaglini et al.](https://arxiv.org/abs/2310.09612) (2023). This includes ResNet-50 & ViT-B/16 architectures that are pretrained on CLIP, ImageNet, or trained from scratch (Random) and fine-tuned on the SQU, ALPH, SHA, or NAT datasets from [Tartaglini et al.](https://arxiv.org/abs/2310.09612) (2023). 

Model weights are stored in the directory `models` as .pth files. The file `models.txt` maps each .pth file to its architecture, pretraining, and fine-tuning dataset. For example, a ViT model that was pretrained on CLIP and fine-tuned on SQU (our best model) has the following entry in `models.txt`:
```
vit,clip,SQU
0de0ddpx
```
This means that this model's weights are stored in the file `models/0de0ddpx.pth`. You don't have to worry about this unless you want to add your own models.

**To evaluate your own model**, first make sure a copy of the weights of your model are stored in the `models` directory. Then, add an entry to the `models.txt` file with the following format:
```
architecture,pretrain,finetune-dataset
filename
```
Make sure this entry is separated from other entries by a new line. You should then be able to run an evaluation on this model for any of the evaluation datasets in `datasets`!

## Running an evaluation
The file `evaluate.py` contains all of the code you need to load a fine-tuned model and evaluate it on a dataset in the `datasets` folder. To run an evaluation, run a command of the following form:
```
python evaluate.py --model_type ARCHITECTURE --pretrain PRETRAIN --ft_dataset FINETUNE-DATASET -ds EVAL-DATASET
```
For example, to evaluate CLIP-pretrained ViT that has been fine-tuned on SQU using the default `filled` dataset, run the following command:
```
python evaluate.py --model_type vit --pretrain clip --ft_dataset SQU -ds filled
```
The model will be loaded and evaluated on `filled`. Model loss and accuracy on the evaluation dataset are computed and printed to the terminal. Loss and accuracy are also stored in the `results` folder as a .csv file. 

By default, the command line arguments take the following values:

<div align="center">

| Command line argument | Possible values |
| - | - |
| `--model_type` | `vit`, `resnet` |
| `--pretrain` | `clip`, `imagenet`, `random` |
| `--ft_dataset` | `SQU`, `ALPH`, `SHA`, `NAT` |

</div>
