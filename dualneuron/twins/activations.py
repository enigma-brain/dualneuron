import torch


def get_layer_by_name(model, layer_name):
    """Get a layer module by its dot-separated name path.
    
    Args:
        model: The model containing the layer
        layer_name: Dot-separated path to the layer (e.g., 'core.features.TaskDriven.layer1.0.bn1')
    
    Returns:
        The layer module
    """
    parts = layer_name.split('.')
    layer = model
    for part in parts:
        layer = getattr(layer, part)
    return layer


class WrapLayer:
    """Captures activations from a specific layer using hooks."""
    
    def __init__(self, model, layer):
        self.model = model
        self.activation = None
        layer = get_layer_by_name(model, layer) if isinstance(layer, str) else layer 
        self.hook = layer.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        self.activation = output
    
    def __call__(self, images):
        """Run forward pass and return captured activation."""
        self.activation = None
        _ = self.model(images)
        return self.activation
    
    def remove(self):
        self.hook.remove()


class ActivationExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.activations = {}
        self.hooks = []
        
    def should_hook(self, module):
        if len(list(module.children())) > 0:
            return False
        
        skip_types = (
            torch.nn.Identity,
            torch.nn.Dropout,
            torch.nn.Dropout2d,
            torch.nn.Dropout3d,
        )
        
        if isinstance(module, skip_types):
            return False
        
        return True
    
    def get_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach()
        return hook
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if self.should_hook(module):
                hook = module.register_forward_hook(self.get_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self, input_tensor):
        self.activations = {}
        with torch.no_grad():
            _ = self.model(input_tensor)
        return self.activations


def count_units(shape):
    """Count units per layer: channels for conv, features for fc"""
    if len(shape) == 4:  # Conv: (batch, channels, H, W)
        return shape[1]
    elif len(shape) == 2:  # FC: (batch, features)
        return shape[1]
    elif len(shape) == 3:  # Transformer: (batch, seq, features)
        return shape[2]
    return None


def extract_center_response(activation, unit_idx):
    """Extract center response for one unit"""
    if len(activation.shape) == 3:  # Conv: (channels, H, W)
        h, w = activation.shape[1], activation.shape[2]
        return activation[unit_idx, h//2, w//2].item()
    elif len(activation.shape) == 1:  # FC: (features,)
        return activation[unit_idx].item()
    elif len(activation.shape) == 2:  # Transformer: (seq, features)
        return activation[0, unit_idx].item()  # class token
    return None


def model_summary(model, input_size=(1, 3, 100, 100)):
    dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
    extractor = ActivationExtractor(model)
    extractor.register_hooks()
    activations = extractor.get_activations(dummy_input)

    total_units = 0
    print(f"\n{'Layer':<60} {'Shape':<25} {'Units'}")
    print('-'*95)

    for name, activation in activations.items():
        shape = tuple(activation.shape)
        num_units = count_units(shape)
        if num_units:
            total_units += num_units
            print(f"{name:<60} {str(shape):<25} {num_units:>8,}")

    print(f"Total units: {total_units:,}")
    extractor.remove_hooks()
    return activations
