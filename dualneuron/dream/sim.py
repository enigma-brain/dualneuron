import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dualneuron.screening.sets import ImagenetImages, RenderedImages
import dualneuron

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dreamsim import dreamsim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.RandomState(123)


def embeddings(
    data_dir, 
    cache_dir="./models",
    output_path=None,
    token=None, 
    split='train', 
    dataset="rendered",
    area='v4',
    use_grayscale=False,
    use_mask=True,
    use_norm=True,
    norm=80.0,
    num_channels=3,
    crop_size=236,
    bg_value=0.0,
    batch_size=32,
    num_workers=0,
):
    """
    Extract DreamSim embeddings from images with mask applied.
    
    Args:
        data_dir: Path to data directory
        cache_dir: Path to cache directory for models
        token: HuggingFace token for ImageNet (if needed)
        split: Dataset split ('train', 'validation', 'test')
        dataset: 'rendered' or 'imagenet'
        bg_value: Background value for masked regions (0.0 = black)
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
    
    Returns:
        embeddings: numpy array of shape (n_images, embedding_dim)
        indices: numpy array of image indices or paths
    """
    assert dataset in ['rendered', 'imagenet']
    
    # Load DreamSim model
    model, _ = dreamsim(
        pretrained=True, 
        device=device, 
        cache_dir=cache_dir
    )
    model = model.eval()
    
    # Load mask the same way as your V1/V4 code
    package_dir = Path(dualneuron.__file__).parent
    model_name = "V4ColorTaskDriven" if area == 'v4' else "V1GrayTaskDriven"
    mask_path = package_dir / "twins" / model_name / "mask.npy"
    mask = np.load(mask_path)
    
    if dataset == "rendered":    
        dset = RenderedImages(
            data_dir=data_dir,
            use_center_crop=True,
            use_resize_output=True,
            use_grayscale=use_grayscale,
            use_normalize=False,
            use_mask=use_mask,
            use_norm=use_norm,
            use_clip=True, 
            mask=mask,
            num_channels=num_channels,
            output_size=(224, 224),
            crop_size=crop_size,
            bg_value=bg_value,
            clip_min=0.0,
            clip_max=1.0,
            norm=norm,
        )
    else:
        dset = ImagenetImages(
            data_dir=data_dir,
            token=token,
            split=split,
            use_center_crop=True,
            use_resize_output=True,
            use_grayscale=use_grayscale,
            use_normalize=False,
            use_mask=use_mask,
            use_norm=use_norm,
            use_clip=True,
            mask=mask,
            num_channels=num_channels,
            output_size=(224, 224),
            crop_size=crop_size,
            bg_value=bg_value,
            clip_min=0.0,
            clip_max=1.0,
            norm=norm,
        )
        
    loader = DataLoader(
        dset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    embeddings_list = []
    
    with torch.no_grad():
        for i, (images, _) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device)
            batch_embeddings = model.embed(images)
            embeddings_list.append(batch_embeddings.cpu().numpy())
            if i==30: break
            
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    
    if output_path is not None:
        np.save(output_path, all_embeddings)
    else:
        return all_embeddings


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract DreamSim embeddings")
    parser.add_argument("--data_dir", type=str, help="Where the data is saved")
    parser.add_argument("--cache_dir", type=str, default="./models", help="Where dreamsim models are cached")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save embeddings .npy file")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token for imagenet")
    parser.add_argument("--split", type=str, default="train", help="train, validation, or test for imagenet")
    parser.add_argument("--dataset", type=str, help="rendered or imagenet")
    parser.add_argument("--area", type=str, default="v4", help="v1 or v4 mask to use")
    parser.add_argument("--use_grayscale", type=bool, default=False, help="Use grayscale images")
    parser.add_argument("--use_mask", type=bool, default=True, help="Whether to use mask")
    parser.add_argument("--use_norm", type=bool, default=True, help="Whether to control norm")
    parser.add_argument("--norm", type=float, default=80.0, help="Norm value if use_norm is True")
    parser.add_argument("--num_channels", type=int, default=3, help="Number of image channels (1 or 3)")
    parser.add_argument("--crop_size", type=int, default=236, help="Crop size for images")
    parser.add_argument("--bg_value", type=float, default=0.0, help="Background value for masked regions (0.0-1.0)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
    args = parser.parse_args()
    
    embeddings_array = embeddings(
        args.data_dir, args.cache_dir, args.output_path, args.token, args.split, args.dataset,
        args.area, args.use_grayscale, args.use_mask, args.use_norm, args.norm,
        args.num_channels, args.crop_size, args.bg_value, args.batch_size, args.num_workers
    )
    