import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dualneuron.screening.sets import ImagenetImages, RenderedImages
from dualneuron.twins.nets import V1GrayTaskDriven, V4ColorTaskDriven
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
    

def activations(
    data_dir, 
    output_dir=None,
    token=None, 
    split='train', 
    dataset="rendered", 
    area='v4',
    layer='neurons',
    ensemble=True,
    batch_size=32,
    num_workers=0,
):
    assert layer in ['neurons', 'features']
    assert area in ['v1', 'v4']
    assert dataset in ['rendered', 'imagenet']
    
    if area == 'v4':
        model = V4ColorTaskDriven(ensemble=ensemble, centered=True)
        if layer == 'neurons':
            neurons = list(range(394))
        else:
            neurons = list(range(1024))
        model_name = "V4ColorTaskDriven"
    else:
        model = V1GrayTaskDriven(ensemble=ensemble, centered=True)
        if layer == 'neurons':
            neurons = list(range(458))
        else:
            neurons = list(range(1024))
        model_name = "V1GrayTaskDriven"

    model = model.eval().to(device)

    package_dir = Path(dualneuron.__file__).parent
    mask_path = package_dir / "twins" / model_name / "mask.npy"
    mask = np.load(mask_path)
       
    if dataset == "rendered":    
        dset = RenderedImages(
            data_dir=data_dir,
            use_center_crop=True,
            use_resize_output=True,
            use_grayscale=False if area == 'v4' else True,
            use_normalize=True,
            use_mask=True,
            use_norm=True,
            use_clip=False,
            mask=mask,
            num_channels=3 if area == 'v4' else 1,
            output_size=(100, 100) if area == 'v4' else (93, 93),
            crop_size=236 if area == 'v4' else 167,
            bg_value=0.0,
            norm=40.0 if area == 'v4' else 12.0
        )
    else:
        dset = ImagenetImages(
            data_dir=data_dir,
            token=token,
            split=split,
            use_center_crop=True,
            use_resize_output=True,
            use_grayscale=False if area == 'v4' else True,
            use_normalize=True,
            use_mask=True,
            use_norm=True,
            use_clip=False,
            mask=mask,
            num_channels=3 if area == 'v4' else 1,
            output_size=(100, 100) if area == 'v4' else (93, 93),
            crop_size=236 if area == 'v4' else 167,
            bg_value=0.0,
            norm=40.0 if area == 'v4' else 12.0,
        )
    
    loader = DataLoader(
        dset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    bsize = loader.batch_size
    outs = torch.zeros(len(loader.dataset), len(neurons))
    
    with torch.no_grad():
        for i, (scenes, _) in tqdm(enumerate(loader), total=len(loader)):
            if layer == 'features':
                output = model.core.features(scenes.cuda())[:, neurons, 3, 3]
            else:
                output = model(scenes.cuda())[:, neurons]

            outs[i*bsize:i*bsize+len(scenes)] = output.cpu().detach()

    resps, idx = torch.sort(outs, dim=0)
    sresps = resps.numpy()
    sidx = idx.numpy()

    if output_dir is None:
        return sresps, sidx
    else:
        endrespdir = os.path.join(output_dir, f"{area}_{layer}_{dataset}_ordered_responses")
        idxdir = os.path.join(output_dir, f"{area}_{layer}_{dataset}_ordered_indices")

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
    parser.add_argument("--area", type=str, default="v4", help="area: v4 or v1")
    parser.add_argument("--layer", type=str, default="neurons", help="neurons or features")
    parser.add_argument("--ensemble", type=bool, default=True, help="use ensemble model")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
    args = parser.parse_args()
    
    activations(
        args.data_dir, args.output_dir, args.token, args.split, args.dataset, 
        args.area, args.layer, args.ensemble, args.batch_size, args.num_workers
    )