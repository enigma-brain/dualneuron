import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dualneuron.screening.sets import ImagenetImages, RenderedImages
from dualneuron.twins.nets import load_model
from dualneuron.twins.activations import get_layer_info, get_spatial_activation
import dualneuron

import os
import numpy as np
import torch

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.RandomState(123)


def screen_activations(
    data_dir,
    output_dir=None,
    token=None,
    split='train',
    dataset="rendered",
    model='v4',
    layer=None,
    location='center',
    ensemble=True,
    batch_size=32,
    num_workers=0,
    device="cuda"
):

    assert dataset in ['rendered', 'imagenet']

    function = load_model(
        architecture=model,
        layer=layer,
        ensemble=ensemble,
        centered=True,
        device=device
    )
    
    package_dir = Path(dualneuron.__file__).parent
    mask_path = package_dir / "twins" / "V4ColorTaskDriven" / "mask.npy"
    mask = np.load(mask_path)
    
    if model == 'v1':
        output_size = (93, 93)
        norm = 12.0
        num_channels = 1
    elif model == 'v4':
        output_size = (100, 100)
        norm = 40.0
        num_channels = 3
    else:
        output_size = (224, 224)
        norm = 80.0
        num_channels = 3
    
    if dataset == "rendered":    
        dset = RenderedImages(
            data_dir=data_dir,
            use_center_crop=True,
            use_resize_output=True,
            use_grayscale=True if model == 'v1' else False,
            use_normalize=True,
            use_mask=True,
            use_norm=True,
            use_clip=False,
            mask=mask,
            num_channels=1 if model == 'v1' else 3,
            output_size=output_size,
            crop_size=167 if model == 'v1' else 236,
            bg_value=0.0,
            norm=norm
        )
    else:
        dset = ImagenetImages(
            data_dir=data_dir,
            token=token,
            split=split,
            use_center_crop=True,
            use_resize_output=True,
            use_grayscale=True if model == 'v1' else False,
            use_normalize=True,
            use_mask=True,
            use_norm=True,
            use_clip=False,
            mask=mask,
            num_channels=1 if model == 'v1' else 3,
            output_size=output_size,
            crop_size=167 if model == 'v1' else 236,
            bg_value=0.0,
            norm=norm,
        )
    
    loader = DataLoader(
        dset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    output_shape, neurons = get_layer_info(
        function, 
        (1, num_channels, *output_size), 
        device=device
    )
    neurons = list(range(neurons))
    
    bsize = loader.batch_size
    outs = torch.zeros(len(loader.dataset), len(neurons))
    
    with torch.no_grad():
        for i, (scenes, _) in tqdm(enumerate(loader), total=len(loader)):
            output = function(scenes.to(device))
            output = get_spatial_activation(
                output, 
                neurons=neurons, 
                location=location
            )
            outs[i*bsize:i*bsize+len(scenes)] = output.cpu().detach()

    resps, idx = torch.sort(outs, dim=0)
    sresps = resps.numpy()
    sidx = idx.numpy()

    if output_dir is None:
        return sresps, sidx
    else:
        endrespdir = os.path.join(
            output_dir, 
            f"{model}_{layer}_{dataset}_ordered_responses"
        )
        idxdir = os.path.join(
            output_dir, 
            f"{model}_{layer}_{dataset}_ordered_indices"
        )

        os.makedirs(endrespdir, exist_ok=True)
        os.makedirs(idxdir, exist_ok=True)

        for i, unit in tqdm(enumerate(neurons), total=len(neurons)):
            np.save(os.path.join(endrespdir, f"{str(unit)}.npy"), sresps[:, i])
            np.save(os.path.join(idxdir, f"{str(unit)}.npy"), sidx[:, i])
        return 
        
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get DERS")
    parser.add_argument("--data_dir", type=str, help="Where the data is saved")
    parser.add_argument("--output_dir", type=str, default=None, help="Where the output is saved")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token for imagenet")
    parser.add_argument("--split", type=str, default="train", help="train, validation, or test")
    parser.add_argument("--dataset", type=str, help="rendered or imagenet")
    parser.add_argument("--model", type=str, default="v4", help="model architecture")
    parser.add_argument("--layer", type=str, default=None, help="layer name, if none use final layer")
    parser.add_argument("--location", default="center", help="spatial location for activation extraction")
    parser.add_argument("--ensemble", type=bool, default=True, help="use ensemble model")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
    parser.add_argument("--device", type=str, default="cuda", help="device to run on")
    args = parser.parse_args()
    
    screen_activations(
        args.data_dir, args.output_dir, args.token, args.split, args.dataset, 
        args.model, args.layer, args.location, args.ensemble, args.batch_size, 
        args.num_workers, args.device
    )