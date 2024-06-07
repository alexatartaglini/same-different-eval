"""
Load and evaluate a fine-tuned same-differnet model on a custom test dataset.

@author: alexatartaglini
"""

import os
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification, logging
import clip
import argparse
import glob
import numpy as np
from sklearn.metrics import accuracy_score
import csv
from tqdm import tqdm

# Turn off warnings from transformers library
logging.set_verbosity_warning()
logging.set_verbosity_error()


# Get device for the models
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except AttributeError:  # if MPS is not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SameDifferentDataset(Dataset):
    """
    This class provides a wrapper for a same-different dataset.
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.im_dict = self.load_dataset(root_dir)
        self.transform = transform

    def __len__(self):
        return len(list(self.im_dict.keys()))

    def __getitem__(self, idx):
        im_path = self.im_dict[idx]['image_path']
        im_path = self.im_dict[idx]['image_path']
        im = Image.open(im_path).convert('RGB')
        label = self.im_dict[idx]['label']

        if self.transform:
            if str(type(self.transform)) == "<class 'torchvision.transforms.transforms.Compose'>":
                item = self.transform(im)
                item = {"image": item, "label": label}
            else:
                item = self.transform.preprocess(np.array(im, dtype=np.float32), return_tensors="pt")
                item["label"] = label

        return item, im_path
    
    def load_dataset(self, root_dir):
        int_to_label = {1: "same", 0: "different"}
        ims = {}
        idx = 0

        for l in int_to_label.keys():
            im_paths = glob.glob(f"{root_dir}/{int_to_label[l]}/*.png")

            for im in im_paths:
                pixels = Image.open(im)
                im_dict = {"image": pixels, "image_path": im, "label": l}
                ims[idx] = im_dict
                idx += 1
                pixels.close()

        return ims

def load_model(model_type, pretrain, ft_dataset):
    """
    Loads a fine-tuned model for evaluation.
    
    The paths to the different models are stored in a text file (models.txt).
    To add your own model, simply append an entry for it to this file :^)

    Parameters
    ----------
    model_type : str
        The type of model architecture to load. ("vit" or "resnet")
    pretrain : str
        The type of pretraining for the model. ("clip", "imagenet", "random")    
    ft_dataset : str
        The same-different dataset that the model has been fine-tuned on 
        (from Tartaglini et al., 2023). ("SQU", "ALPH", "SHA", or "NAT")

    Returns
    -------
    model : The model with the fine-tuned weights loaded.
    transform : The appropriate image transform for the model.
    """
    
    # Open models.txt and find the file name corresponding to the requested model type
    mf = open("models.txt", "r").readlines()
    return_path = False

    for line in mf:
        line = line.replace("\n", "").split(",")
        
        if return_path:  # We are at the line containing the model path name
            model_path = line[0]
            break
        
        if len(line) > 1:
            if line[0] == model_type and line[1] == pretrain and line[2].upper() == ft_dataset:
                return_path = True  # We are at the line before the model path name
                
    if not return_path:
        raise FileNotFoundError(f"Model {model_type},{pretrain},{ft_dataset} is not defined in models.txt.")

    if not os.path.exists(f"models/{model_path}.pth"):
        raise FileNotFoundError(f"Model file {model_path}.pth does not exist or is not in the models/ directory.")
    
    ckpt = torch.load(f"models/{model_path}.pth", map_location=device)  # Load the model weights
    
    if pretrain != "clip":
        if model_type == "resnet":
            model = models.resnet50(pretrained=False)
            
            # Replace the final layer to output logprobs for two classes ("same" and "different")
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 2)
    
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
        elif model_type == "vit":
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                num_labels=2,
            )
            transform = ViTImageProcessor(do_resize=False).from_pretrained("google/vit-base-patch16-224-in21k")
    else:                
        if model_type == "vit":
            model, transform = clip.load("ViT-B/16", device=device)
            in_features = model.visual.proj.shape[1]
            
        else:
            model, transform = clip.load("RN50", device=device)
            in_features = model.visual.output_dim
        
        # Replace the final layer to output logprobs for two classes ("same" and "different")
        fc = torch.nn.Linear(in_features, 2).to(device)
        model = torch.nn.Sequential(model.visual, fc).float()
    
    model = model.to(device)  # Move model to device
    model.load_state_dict(ckpt)  # Load weights into model
    model.eval()  # Put model in evaluation mode
    
    return model, transform

def run_eval(model, eval_dataloader, model_type="vit", pretrain="clip"):
    """
    Compute model loss and accuracy on an evaluation dataset.

    Parameters
    ----------
    model : ViTForImageClassification, torchvision model, or clip model
        The loaded model to run the evaluation on.
    eval_dataloader : DataLoader
        The DataLoader for the evaluation dataset.
    model_type : str, optional
        The model architecture ("vit" or "resnet"). The default is "vit".
    pretrain : str, optional
        The model pretraining ("clip", "imagenet", or "random"). The default is "clip".

    Returns
    -------
    metric_dict : dict
        A dictionary containing model loss and accuracy on the eval dataset.
    """
    
    criterion = torch.nn.CrossEntropyLoss()  # Loss criterion
    metric_dict = {}  # Dictionary to store results
    
    with torch.no_grad():  # Run evaluation without gradient updates to model
        # Initialize loss & accuracy at 0
        running_loss = 0.0  
        running_acc = 0.0

        # Enumerate over image batches
        # bi = batch index, d = "item" from SameDifferentDataset __getitem__ method, 
        # f = "im_path" from SameDifferentDataset __getitem__ method.
        for bi, (d, f) in enumerate(tqdm(eval_dataloader)):
            # The dataloader automatically passes images through the model's transform,
            # so now we need to get the image from the transform
            if model_type == "vit" and pretrain != "clip":
                inputs = d["pixel_values"].squeeze(1).to(device)
            else:
                inputs = d["image"].to(device)
            
            # Get the same/different labels for the images
            labels = d["label"].to(device)

            # Pass the images through the model
            outputs = model(inputs)
            if model_type == "vit" and pretrain != "clip":
                outputs = outputs.logits

            loss = criterion(outputs, labels)  # Compute loss
            preds = outputs.argmax(1)  # Get model decision ("same" or "different")
            acc = accuracy_score(labels.to("cpu"), preds.to("cpu"))  # Compute accuracy

            running_acc += acc * inputs.size(0)
            running_loss += loss.detach().item() * inputs.size(0)

        epoch_loss = running_loss / len(eval_dataloader.dataset)
        epoch_acc = running_acc / len(eval_dataloader.dataset)

        metric_dict["loss"] = epoch_loss
        metric_dict["acc"] = epoch_acc
    
    return metric_dict


if __name__ == "__main__":
    # Define input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="Model architecture to evaluate ('vit' or 'resnet').",
                        type=str, required=True)
    parser.add_argument("--pretrain", help="Model pretraining to evaluate ('clip', 'imagenet', or 'random').",
                        type=str, required=True)
    parser.add_argument("--ft_dataset", help="Fine-tuning dataset to evaluate ('SQU', 'ALPH', 'SHA', or 'NAT').",
                        type=str, required=True)
    parser.add_argument("-ds", "--eval_dataset", help="Name of evaluation dataset",
                        type=str, required=False, default="filled")
    args = parser.parse_args()
    
    args.ft_dataset = args.ft_dataset.upper()
    
    # Check input arguments
    assert args.model_type in ["vit", "resnet"], "Argument --model_type must be 'vit' or 'resnet'."
    assert args.pretrain in ["clip", "imagenet", "random"], "Argument --pretrain must be 'clip', 'imagenet', or 'random'."
    assert args.ft_dataset in ["SQU", "ALPH", "SHA", "NAT"], "Argument --ft_dataset must be 'SQU', 'ALPH', 'SHA', or 'NAT'."
    
    if not os.path.exists(f"datasets/{args.eval_dataset}"):
        raise FileNotFoundError("Eval dataset {args.eval_dataset} does not exist or is not in the datasets/ directory.")
    
    # Load model & model transform
    print("\nLoading model...")
    model, transform = load_model(args.model_type, args.pretrain, args.ft_dataset)
    
    print("Loading evaluation dataset...")
    eval_dataset = SameDifferentDataset(f"datasets/{args.eval_dataset}", transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1024, shuffle=True)
    
    print("Running evaluation...")
    results = run_eval(model, eval_dataloader, model_type=args.model_type, pretrain=args.pretrain)  # Get model loss & acc

    print("\n--------------------------------")
    print(f"RESULTS on {args.eval_dataset}:")
    print(f"\tLoss: {results['loss']}")
    print(f"\tAccuracy: {results['acc']}")
    print("--------------------------------")
    
    # Store model results
    result_path = f"{args.model_type}_{args.pretrain}_{args.ft_dataset}"
    os.makedirs(f"results/{result_path}", exist_ok=True)
    
    with open(f"results/{result_path}/{args.eval_dataset}.csv", "w") as result_file:  
        writer = csv.writer(result_file)
        writer.writerow(["Metric", "Value"])
        
        for key, value in results.items():
           writer.writerow([key, value])
  