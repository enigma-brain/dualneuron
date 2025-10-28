import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
from tqdm import tqdm
from dualneuron.twins.nets import V1GrayTaskDriven, V4ColorTaskDriven
from dualneuron.synthesis.ascend import pixel_ascending, fourier_ascending
import torch
from pathlib import Path


class DualNeuronDataset:
    def __init__(self, hdf5_path):
        self.path = Path(hdf5_path)
        
    def get_neuron_results(
        self, 
        neuron_id, seed=None, 
        activation_type='most'
    ):
        with h5py.File(self.path, 'r') as f:
            if seed is None:
                # Return all seeds
                results = []
                for seed_group in f[f'neuron_{neuron_id:04d}'].keys():
                    data = f[f'neuron_{neuron_id:04d}/{seed_group}/{activation_type}']
                    results.append({
                        'image': data['image'][:],
                        'alpha': data['alpha'][:],
                        'activations': data['activations'][:]
                    })
                return results
            else:
                data = f[f'neuron_{neuron_id:04d}/seed_{seed}/{activation_type}']
                return {
                    'image': data['image'][:],
                    'alpha': data['alpha'][:],
                    'activations': data['activations'][:]
                }


def generate_poles(output_dir=None, num_seeds=5, v1_neurons=458, v4_neurons=394):
    seeds = np.random.choice(10000, size=num_seeds, replace=False).tolist()
    models = {}
    
    if v1_neurons is not None:
        if isinstance(v1_neurons, int):
            v1_neurons = range(v1_neurons)
        models['V1GrayTaskDriven'] = (V1GrayTaskDriven, v1_neurons)
    
    if v4_neurons is not None:
        if isinstance(v4_neurons, int):
            v4_neurons = range(v4_neurons)
        models['V4ColorTaskDriven'] = (V4ColorTaskDriven, v4_neurons)
    
    if not models:
        print("No models to process. Both v1_neurons and v4_neurons are None.")
        return

    ascend = {
        'V1GrayTaskDriven': pixel_ascending,
        'V4ColorTaskDriven': fourier_ascending
    }
    
    v1_params = {
        'image_size': 93,
        'channels': 1,
        'total_steps': 128,
        'learning_rate': 0.05,
        'lr_schedule': True,
        'noise': 0.05,
        'values_range': (-1.0, 1.0),
        'nb_crops': 4,
        'box_size': (1.0, 1.0),
        'target_norm': 12.0,
        'init_std': 0.05,
        'jitter_std': 0.05,
        'oversample': 1,
        'reflect_pad_frac': 0.05,
        'device': 'cuda',
        'verbose': False
    }

    v4_params = {
        'magnitude_path': 'natural_rgb.npy',
        'total_steps': 128,
        'learning_rate': 1.0,
        'lr_schedule': True,
        'noise': 0.08,
        'values_range': (-2.0, 2.0),
        'range_fn': 'sigmoid',
        'nb_crops': 4,
        'box_size': (1.0, 1.0),
        'target_norm': 40.0,
        'jitter_std': 0.05,
        'oversample': 1,
        'reflect_pad_frac': 0.05,
        'device': 'cuda',
        'verbose': False
    }

    output_path = Path(output_dir) if output_dir is not None else Path.cwd()

    for model_name, (model_class, n_neurons) in models.items():
        output_file = output_path / f'{model_name.lower()}_results.h5'
        print(f"Generating Output file: {output_file}")
        
        function = model_class(centered=True, ensemble=True).eval().to('cuda')
        params = v1_params if model_name == 'V1GrayTaskDriven' else v4_params
        
        with h5py.File(output_file, 'w') as f:
            for neuron_id in tqdm(n_neurons):
                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    
                    for weight_name, weight in [('least', -1), ('most', 1)]:
                        image, alpha, act = ascend[model_name](
                            lambda images: weight * torch.mean(function(images)[:, neuron_id]),
                            **params
                        )
                        
                        if torch.is_tensor(image):
                            image = image.cpu().numpy()
                        if torch.is_tensor(alpha):
                            alpha = alpha.cpu().numpy()
                        if torch.is_tensor(act):
                            act = act.cpu().numpy()
                            
                        group_path = f'neuron_{neuron_id:04d}/seed_{seed}/{weight_name}'
                        group = f.create_group(group_path)
                        group.create_dataset(
                            'image', 
                            data=image, 
                            compression='gzip', 
                            compression_opts=4
                        )
                        group.create_dataset(
                            'alpha', 
                            data=alpha, 
                            compression='gzip', 
                            compression_opts=4
                        )
                        group.create_dataset('activations', data=act)
                        group.attrs.update({
                            'model': model_name,
                            'neuron_id': neuron_id,
                            'seed': seed,
                            'weight': weight,
                            'type': weight_name
                        })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate dual neuron poles dataset')
    parser.add_argument(
        '--output-dir', '-o', 
        type=str, default=None,
        help='Output directory for HDF5 files (default: current directory)'
    )
    parser.add_argument(
        '--num-seeds', '-n',
        type=int, default=5,
        help='Number of random seeds to use (default: 5)'
    )
    parser.add_argument(
        '--v1-neurons', '-v1',
        type=int, default=458,
        help='Number of V1 neurons to generate (default: 458, use 0 to skip)'
    )
    parser.add_argument(
        '--v4-neurons', '-v4',
        type=int, default=394,
        help='Number of V4 neurons to generate (default: 394, use 0 to skip)'
    )

    args = parser.parse_args()
    v1 = args.v1_neurons if args.v1_neurons > 0 else None
    v4 = args.v4_neurons if args.v4_neurons > 0 else None
    
    generate_poles(
        output_dir=args.output_dir,
        num_seeds=args.num_seeds,
        v1_neurons=v1,
        v4_neurons=v4
    )