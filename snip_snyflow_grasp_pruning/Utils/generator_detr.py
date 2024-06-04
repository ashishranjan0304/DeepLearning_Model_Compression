def get_masks(module):
    """Returns an iterator over module masks, yielding the mask."""
    if hasattr(module, 'mask'):
        yield module.mask
 
def get_masked_parameters(model):
    """Returns an iterator over model's prunable parameters, yielding both the mask and parameter tensors."""
    for module in model.modules():
        if hasattr(module, 'mask'):
            for mask, param in zip(get_masks(module), module.parameters(recurse=False)):
                if param.requires_grad:  # Only consider trainable parameters
                    print(f"Module: {module.__class__.__name__}, Mask shape: {mask.shape}, Param shape: {param.shape}")
                    yield mask, param
                else:
                    print(f"Module: {module.__class__.__name__} has a non-trainable parameter with shape {param.shape}")

def print_masked_parameters_info(model):
    """Prints the number of parameters and masks module-wise."""
    for module in model.modules():
        if hasattr(module, 'mask'):
            param_count = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            mask_count = sum(m.numel() for m in get_masks(module))
            print(f"Module: {module.__class__.__name__}, Trainable Parameters: {param_count}, Masks: {mask_count}")
