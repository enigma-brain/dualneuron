import torch


def count_units(shape):
    """
    Count the number of feature units in a layer output.
    
    Determines the number of independent features based on tensor shape:
    convolutional layers have channels, fully connected layers have features,
    and transformer layers have embedding dimensions.
    
    Args:
        shape (tuple): Shape of the layer output tensor.
    
    Returns:
        int or None: Number of units, or None if shape is unrecognized.
            - Conv (batch, channels, H, W): returns channels
            - FC (batch, features): returns features
            - Transformer (batch, seq, features): returns features
    """
    if len(shape) == 4: # Conv: (batch, channels, H, W)
        return shape[1]
    elif len(shape) == 2: # FC: (batch, features)
        return shape[1]
    elif len(shape) == 3: # Transformer: (batch, seq, features)
        return shape[2]
    return None


def get_layer_info(function, input_shape, device="cuda"):
    """
    Get output shape and neuron count from a model by running a forward pass.
    
    Args:
        function (callable): Model or layer wrapper that accepts tensor input.
        input_shape (tuple): Input tensor shape as (batch, channels, height, width).
        device (str or torch.device): Device to run forward pass on. Default: "cuda".
    
    Returns:
        tuple: (output_shape, num_neurons)
            - output_shape (torch.Size): Shape of the layer output.
            - num_neurons (int): Number of feature units in the output.
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
    
    Handles different tensor formats (conv, FC, transformer) and provides
    flexible spatial/sequence position selection.
    
    Args:
        output (torch.Tensor): Layer output tensor. Supported shapes:
            - Conv: (batch, channels, H, W)
            - FC: (batch, features)
            - Transformer: (batch, seq, features)
        neurons (list, np.ndarray, or None): Indices of neurons to extract.
            If None, returns all neurons. Default: None.
        location: Spatial or sequence position to extract. Interpretation
            depends on tensor type:
            
            For Conv layers (4D):
                - None: Average over spatial dimensions (global average pool)
                - 'center': Extract from center pixel (H//2, W//2)
                - (h, w) tuple: Extract from specific spatial location
            
            For Transformer layers (3D):
                - None: Average over sequence dimension
                - 'center': Extract middle token (seq_len // 2)
                - int: Extract specific token position (e.g., 0 for CLS)
            
            For FC layers (2D):
                - Ignored (no spatial dimension)
    
    Returns:
        torch.Tensor: Extracted activations with shape (batch, num_neurons).
    
    Raises:
        ValueError: If location format is invalid for the tensor type,
            or if output shape is unrecognized.
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
    """
    Extract activations from intermediate layers of a neural network.
    
    Uses PyTorch forward hooks to capture layer outputs during inference.
    Supports two modes:
        1. Targeted extraction: Efficiently extract from a single specified layer
        2. Full extraction: Capture all layer activations in one forward pass
    
    Args:
        model (torch.nn.Module): The neural network model.
        layer (str, optional): Target layer name for extraction. If None,
            use get_all_activations() to extract from all layers.
    
    Attributes:
        model (torch.nn.Module): The wrapped model.
        layer (str): Current target layer name.
        activations (dict): Stored activations from last forward pass.
        hooks (list): Registered forward hook handles.
        layer_counts (dict): Tracks repeated layer names for unique IDs.
    
    Note:
        Layer names follow PyTorch's named_modules() convention. Use
        model_summary() to see available layer names and shapes.
    """
    def __init__(self, model, layer=None):
        self.model = model
        self.layer = layer
        self.activations = {}
        self.hooks = []
        self.layer_counts = {}
        
        if layer is not None:
            self.register_hooks(target_only=True)

    def get_base_name(self, full_name):
        """
        Extract base layer name without occurrence suffix.
        
        Handles layer names like 'relu_0', 'relu_1' that arise from
        reused modules (e.g., shared ReLU layers).
        
        Args:
            full_name (str): Full layer name, possibly with '_N' suffix.
        
        Returns:
            str: Base name without the occurrence counter.
        """
        if "_" in full_name and full_name.split('_')[-1].isdigit():
            return full_name.rsplit('_', 1)[0]
        return full_name

    def hook_factory(self, name):
        """
        Create a forward hook function for a specific layer.
        
        Handles naming collisions from reused modules (e.g., shared ReLU)
        by appending occurrence counters to layer names.
        
        Args:
            name (str): Base name of the layer.
        
        Returns:
            callable: Hook function compatible with register_forward_hook().
        
        Note:
            The hook stores activations differently based on mode:
            - Full mode (self.layer is None): Stores under unique layer name
            - Targeted mode: Stores under 'result' key for efficient retrieval
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
        Register forward hooks on model layers.
        
        Args:
            target_only (bool): If True and self.layer is set, only hook
                the target layer's module for efficiency. If False, hooks
                all leaf modules. Default: False.
        
        Note:
            Automatically removes existing hooks before registering new ones.
            Only hooks leaf modules (those without children).
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
        """
        Remove all registered forward hooks.
        
        Should be called when done with extraction to prevent memory leaks,
        or automatically called before registering new hooks.
        """
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.layer_counts = {}

    def get_all_activations(self, input_tensor):
        """
        Extract activations from all layers in a single forward pass.
        
        Useful for model exploration, debugging, or when multiple layer
        activations are needed simultaneously.
        
        Args:
            input_tensor (torch.Tensor): Input to the model.
        
        Returns:
            dict: Mapping from layer names to activation tensors.
                Keys are unique names (e.g., 'layer3.relu', 'layer3.relu_1').
                Values are detached tensors.
        
        Note:
            This method temporarily switches to full extraction mode,
            which may be slower than targeted extraction for single layers.
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
        Extract activation from the target layer.
        
        Runs a forward pass and returns the activation tensor for the
        specified layer. More efficient than get_all_activations() when
        only one layer is needed.
        
        Args:
            input_tensor (torch.Tensor): Input to the model.
            layer (str, optional): Layer name to extract from. If provided,
                updates the target layer and re-registers hooks.
        
        Returns:
            torch.Tensor: Activation tensor from the target layer.
        
        Raises:
            ValueError: If no target layer is specified (use get_all_activations()
                instead).
                
        Note:
            Unlike get_all_activations(), this returns the raw tensor with
            gradients enabled (not detached), allowing gradient computation
            for activation maximization.
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