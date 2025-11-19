import torch


def count_units(shape):
    """Count units per layer: channels for conv, features for fc"""
    if len(shape) == 4: # Conv: (batch, channels, H, W)
        return shape[1]
    elif len(shape) == 2: # FC: (batch, features)
        return shape[1]
    elif len(shape) == 3: # Transformer: (batch, seq, features)
        return shape[2]
    return None


def get_layer_info(function, input_shape, device="cuda"):
    """
    Get the shape and number of neurons 
    from a layer by doing a forward pass.
    
    Args:
        function: The model/layer wrapper
        input_shape: Tuple of (batch, channels, height, width)
        device: Device to run on
    
    Returns:
        Tuple of (num_neurons, spatial_height, spatial_width) 
        or (num_neurons,) for FC layers
    """
    dummy_input = torch.randn(1, *input_shape[1:]).to(device)
    with torch.no_grad():
        output = function(dummy_input)
    
    shape = output.shape
    num_neurons = count_units(shape)
    return shape, num_neurons


def get_spatial_activation(output, neurons=None, location=None):
    """
    Extract activations from specified neurons and spatial location.
    
    Args:
        output: Tensor of different shapes:
            - Conv: (batch, channels, H, W)
            - FC: (batch, features)
            - Transformer: (batch, seq, features)
        neurons: List/array of neuron indices, or None for all neurons
        location: Tuple of (h, w) or 'center' or None or int (for seq position)
            - For Conv layers:
                - (h, w): specific spatial location
                - 'center': center of spatial dimensions
                - None: average over spatial dimensions
            - For Transformer layers:
                - int: specific sequence position (e.g., 0 for CLS token)
                - 'center': middle token in the sequence after CLS
                - None: average over sequence dimension
    
    Returns:
        Tensor of extracted activations
    """
    if len(output.shape) == 4:  # Conv layer: (batch, channels, H, W)
        if location is None:
            # Average pool over spatial dimensions
            output = output.mean(dim=(2, 3))
        elif location == 'center':
            h, w = output.shape[2], output.shape[3]
            output = output[:, :, h//2, w//2]
        elif isinstance(location, (tuple, list)) and len(location) == 2:
            h, w = location
            output = output[:, :, h, w]
        else:
            raise ValueError(f"Invalid location for Conv layer: {location}")
        
        if neurons is not None:
            output = output[:, neurons]
            
    elif len(output.shape) == 3:  # Transformer: (batch, seq, features)
        # Handle sequence position selection first
        if location is None:
            # Average over sequence dimension
            output = output.mean(dim=1)  # (batch, features)
        elif location == 'center':
            output = output[:, output.shape[1] // 2, :]  # (batch, features)
        elif isinstance(location, int):
            output = output[:, location, :]  # (batch, features)
        else:
            raise ValueError(f"Invalid location for Transformer layer: {location}")
        
        if neurons is not None:
            output = output[:, neurons]
            
    elif len(output.shape) == 2:  # FC layer: (batch, features)
        # Handle neuron selection
        if neurons is not None:
            output = output[:, neurons]
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")
    
    return output


class ActivationExtractor:
    def __init__(self, model, layer=None):
        self.model = model
        self.layer = layer
        self.activations = {}
        self.hooks = []
        self.layer_counts = {}
        
        if layer is not None:
            self.register_hooks(target_only=True)

    def get_base_name(self, full_name):
        if "_" in full_name and full_name.split('_')[-1].isdigit():
            return full_name.rsplit('_', 1)[0]
        return full_name

    def hook_factory(self, name):
        """
        Creates a hook that handles naming 
        collisions (e.g. reused Relus)
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            
            # 1. Generate the controlled name (unique ID)
            count = self.layer_counts.get(name, 0)
            unique_name = f"{name}_{count}" if count > 0 else name
            self.layer_counts[name] = count + 1
            
            # 2. Store activation depending on mode
            if self.layer is None:
                self.activations[unique_name] = output.detach()
            
            # Mode B: Extract ONE (Targeted)
            elif unique_name == self.layer:
                self.activations['result'] = output
                
        return hook

    def register_hooks(self, target_only=False):
        """
        Registers hooks.
        Optimization: If target_only is True, we only hook the specific module 
        """
        self.remove_hooks()
        
        target_base = None
        if target_only and self.layer:
            target_base = self.get_base_name(self.layer)

        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue
            
            if target_only and target_base and name != target_base:
                continue

            self.hooks.append(module.register_forward_hook(self.hook_factory(name)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.layer_counts = {}

    def get_all_activations(self, input_tensor):
        """
        Runs a pass and returns a dictionary of {name: tensor} for EVERY layer.
        Replaces the old ActivationExtractor.get_activations()
        """
        self.remove_hooks()
        self.layer = None
        self.register_hooks(target_only=False)
        
        self.layer_counts = {}
        self.activations = {}
        
        with torch.no_grad():
            self.model(input_tensor)
            
        return self.activations

    def __call__(self, input_tensor, layer=None):
        """
        Run the model and return the specific activation for ONE layer.
        """
        if layer is not None and layer != self.layer:
            self.layer = layer
            self.register_hooks(target_only=True)
            
        if self.layer is None:
            raise ValueError("No target layer specified. Use .get_all_activations().")

        self.layer_counts = {}
        self.activations = {}
        
        _ = self.model(input_tensor)
        
        return self.activations.get('result', None)