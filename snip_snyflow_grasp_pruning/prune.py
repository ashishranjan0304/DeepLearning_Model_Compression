from tqdm import tqdm
import torch
import numpy as np


def prune_loop(model, loss, pruner, dataloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
            print(sparse)
            print("sparsity value")
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        # Invert scores
        if invert:
            pruner.invert()
        pruner.mask(sparse, scope)

        # Calculate sparsity after each epoch
        remaining_params, total_params = pruner.stats()
        print(f"Epoch {epoch + 1}/{epochs}, Sparsity: {remaining_params}/{total_params} ({remaining_params / total_params:.4f})")

    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    print("ashish")
    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    print("{} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
    if np.abs(remaining_params - total_params*sparsity) >= 5:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))