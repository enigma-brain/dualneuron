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
    token=None, 
    split='train', 
    dataset="rendered",
    area='v4',
    bg_value=0.0,
    batch_size=32,
    num_workers=0,
):
    """
    Extract DreamSim embeddings from images with mask applied.
    
    Args:
        data_dir: Path to data directory
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
    model, _ = dreamsim(pretrained=True, device=device)
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
            use_grayscale=False if area == 'v4' else True,
            use_normalize=False,
            use_mask=True,
            use_norm=True,
            mask=mask,
            num_channels=3 if area == 'v4' else 1,
            output_size=(224, 224),
            crop_size=236 if area == 'v4' else 167,
            bg_value=bg_value,
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
            use_normalize=False,
            use_mask=True,
            use_norm=True,
            mask=mask,
            num_channels=3 if area == 'v4' else 1,
            output_size=(224, 224),
            crop_size=236 if area == 'v4' else 167,
            bg_value=bg_value,
            norm=40.0 if area == 'v4' else 12.0,
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
            
    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    return all_embeddings


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract DreamSim embeddings")
    parser.add_argument("--data_dir", type=str, help="Where the data is saved")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token for imagenet")
    parser.add_argument("--split", type=str, default="train", help="train, validation, or test")
    parser.add_argument("--dataset", type=str, help="rendered or imagenet")
    parser.add_argument("--model_type", type=str, default="ensemble", 
                       help="ensemble, dino_vitb16, clip_vitb32, or open_clip_vitb32")
    parser.add_argument("--bg_value", type=float, default=0.0, 
                       help="Background value for masked regions (0.0-1.0)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Path to save embeddings .npy file")
    args = parser.parse_args()
    
    embeddings_array, indices = embeddings(
        args.data_dir, args.token, args.split, args.dataset, 
        args.model_type, args.bg_value, args.batch_size, args.num_workers
    )
    
    # Save embeddings
    np.save(args.output_path, embeddings_array)
    indices_path = args.output_path.replace('.npy', '_indices.npy')
    np.save(indices_path, indices)
    
    print(f"Saved embeddings to: {args.output_path}")
    print(f"Saved indices to: {indices_path}")