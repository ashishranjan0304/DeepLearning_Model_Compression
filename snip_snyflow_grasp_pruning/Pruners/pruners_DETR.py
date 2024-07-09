import torch
import numpy as np
import sys
sys.path.append('../../DETR-Object-Detection/detr')
import util.misc as utils

class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf 
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        print("globalscore")
        print(global_scores.shape)
        k = int((1.0 - sparsity) * global_scores.numel())
        print(k)
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
    
    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0 
        for mask, _ in self.masked_parameters:
             remaining_params += mask.detach().cpu().numpy().sum()
             total_params += mask.numel()
        return remaining_params, total_params


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)
    
    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, criterion, dataloader, device):
        # Allow masks to have gradient
        for mask, param in self.masked_parameters:
            mask.requires_grad = True

        # Ensure model is in train mode
        model.train()
        criterion.train()

        # Metric logger for debugging
        metric_logger = utils.MetricLogger(delimiter="  ")

        for samples, targets in metric_logger.log_every(dataloader, 10, 'SNIP Scoring:'):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Backward pass
            model.zero_grad()
            loss.backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, criterion, dataloader, device):
        # Ensure masks have gradients enabled
        for mask, param in self.masked_parameters:
            mask.requires_grad = True

        # Ensure model is in train mode
        model.train()
        criterion.train()

        # Metric logger for debugging
        metric_logger = utils.MetricLogger(delimiter="  ")

        # First gradient vector without computational graph
        stopped_grads = 0
        for samples, targets in metric_logger.log_every(dataloader, 10, 'GraSP First Pass:'):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            grads = torch.autograd.grad(loss, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # Second gradient vector with computational graph
        for samples, targets in metric_logger.log_every(dataloader, 10, 'GraSP Second Pass:'):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            grads = torch.autograd.grad(loss, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()
        
        # Calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # Normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)

        # Disable gradients for masks
        for mask, param in self.masked_parameters:
            mask.requires_grad = False



import torch
from torch import nn
from typing import List

class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, criterion, dataloader, device):
        @torch.no_grad()
        def linearize(model):
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        # Linearize the model
        signs = linearize(model)

        # Dummy input for the forward pass
        dummy_input = nested_tensor_from_tensor_list([torch.ones((3, 400, 400)).to(device)])  # Adjust input shape as needed
        model(dummy_input)

        # Compute SynFlow scores
        for mask, param in self.masked_parameters:
            param.grad = None

        for name, param in model.named_parameters():
            if param.requires_grad and id(param) in [id(p) for _, p in self.masked_parameters]:
                param.grad = None
                objective = param.abs().sum()
                objective.backward(retain_graph=True)
                self.scores[id(param)] = (param.grad * param).abs().detach()
                param.grad = None

        # Non-linearize the model
        nonlinearize(model, signs)

        # Normalize scores
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for mask, param in self.masked_parameters:
            self.scores[id(param)].div_(norm)
            mask.requires_grad = False

        # Zero out the gradients for all parameters
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

def _max_by_axis(the_list):
    # Helper function to get the maximum size along each dimension
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            if item > maxes[index]:
                maxes[index] = item
    return maxes

# Helper function to create a nested tensor from a list of tensors
def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

import torch
from torch import nn

class SynSNIP(Pruner):
    def __init__(self, masked_parameters, alpha=0.5):
        super(SynSNIP, self).__init__(masked_parameters)
        self.alpha = alpha

    def score(self, model, criterion, dataloader, device):
        @torch.no_grad()
        def linearize(model):
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        # Linearize the model
        signs = linearize(model)

        # Dummy input for the forward pass
        dummy_input = nested_tensor_from_tensor_list([torch.ones((3, 400, 400)).to(device)])
        model(dummy_input)

        # Compute initial gradient-based scores (SNIP part)
        for mask, param in self.masked_parameters:
            mask.requires_grad = True

        model.train()
        criterion.train()

        metric_logger = utils.MetricLogger(delimiter="  ")

        for samples, targets in metric_logger.log_every(dataloader, 10, 'SynSNIP First Pass:'):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            model.zero_grad()
            loss.backward()

            for mask, param in self.masked_parameters:
                self.scores[id(param)] = self.alpha * torch.clone(mask.grad).detach().abs_()

        nonlinearize(model, signs)

        # Compute structural importance scores (SynFlow part)
        for mask, param in self.masked_parameters:
            param.grad = None

        for name, param in model.named_parameters():
            if param.requires_grad and id(param) in [id(p) for _, p in self.masked_parameters]:
                param.grad = None
                objective = param.abs().sum()
                objective.backward(retain_graph=True)
                if id(param) in self.scores:
                    self.scores[id(param)] += (1 - self.alpha) * (param.grad * param).abs().detach()
                else:
                    self.scores[id(param)] = (1 - self.alpha) * (param.grad * param).abs().detach()
                param.grad = None

        # Normalize scores
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for mask, param in self.masked_parameters:
            self.scores[id(param)].div_(norm)
            mask.requires_grad = False

        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()


import torch

class OurAlgo(Pruner):
    def __init__(self, masked_parameters, mag_weight=0.5, snip_weight=0.5):
        super(OurAlgo, self).__init__(masked_parameters)
        self.mag_weight = mag_weight
        self.snip_weight = snip_weight

    def score(self, model, criterion, dataloader, device):
        mag_scores = {}
        snip_scores = {}

        # Calculate Magnitude scores
        for _, p in self.masked_parameters:
            mag_scores[id(p)] = torch.clone(p.data).detach().abs_()

        # Allow masks to have gradient
        for mask, param in self.masked_parameters:
            mask.requires_grad = True

        # Ensure model is in train mode
        model.train()
        criterion.train()

        # Metric logger for debugging
        metric_logger = utils.MetricLogger(delimiter="  ")

        for samples, targets in metric_logger.log_every(dataloader, 10, 'SNIP Scoring:'):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Backward pass
            model.zero_grad()
            loss.backward()

        # Calculate SNIP scores |g * theta|
        for m, p in self.masked_parameters:
            snip_scores[id(p)] = torch.clone(m.grad).detach().abs_() * torch.clone(p.data).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # Normalize SNIP scores
        all_snip_scores = torch.cat([torch.flatten(v) for v in snip_scores.values()])
        snip_norm = torch.sum(all_snip_scores)
        for key in snip_scores:
            snip_scores[key].div_(snip_norm)

        # Combine Magnitude and SNIP scores using weighted sum
        for _, p in self.masked_parameters:
            combined_score = self.mag_weight * mag_scores[id(p)] + self.snip_weight * snip_scores[id(p)]
            self.scores[id(p)] = combined_score

        # Normalize combined scores
        all_combined_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        combined_norm = torch.sum(all_combined_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(combined_norm)

class MagSNIPnorm(Pruner):
    def __init__(self, masked_parameters, mag_weight=0.2, snip_weight=0.8):
        super(MagSNIPnorm, self).__init__(masked_parameters)
        self.mag_weight = mag_weight
        self.snip_weight = snip_weight

    def score(self, model, criterion, dataloader, device):
        mag_scores = {}
        snip_scores = {}

        # Calculate Magnitude scores
        for _, p in self.masked_parameters:
            mag_scores[id(p)] = torch.clone(p.data).detach().abs_()

        # Normalize Magnitude scores
        all_mag_scores = torch.cat([torch.flatten(v) for v in mag_scores.values()])
        mag_norm = torch.sum(all_mag_scores)
        for key in mag_scores:
            mag_scores[key].div_(mag_norm)

        # Allow masks to have gradient
        for mask, param in self.masked_parameters:
            mask.requires_grad = True

        # Ensure model is in train mode
        model.train()
        criterion.train()

        # Metric logger for debugging
        metric_logger = utils.MetricLogger(delimiter="  ")

        for samples, targets in metric_logger.log_every(dataloader, 10, 'SNIP Scoring:'):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Backward pass
            model.zero_grad()
            loss.backward()

        # Calculate SNIP scores |g * theta|
        for m, p in self.masked_parameters:
            snip_scores[id(p)] = torch.clone(m.grad).detach().abs_() * torch.clone(p.data).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # Normalize SNIP scores
        all_snip_scores = torch.cat([torch.flatten(v) for v in snip_scores.values()])
        snip_norm = torch.sum(all_snip_scores)
        for key in snip_scores:
            snip_scores[key].div_(snip_norm)

        # Combine normalized Magnitude and SNIP scores using weighted sum
        for _, p in self.masked_parameters:
            combined_score = self.mag_weight * mag_scores[id(p)] + self.snip_weight * snip_scores[id(p)]
            self.scores[id(p)] = combined_score

        # Normalize combined scores (optional, can be beneficial depending on usage context)
        all_combined_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        combined_norm = torch.sum(all_combined_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(combined_norm)



class MagSNIPvar(Pruner):
    def __init__(self, masked_parameters, mag_weight=0.33, snip_weight=0.33, var_weight=0.33):
        super(MagSNIPvar, self).__init__(masked_parameters)
        self.mag_weight = mag_weight
        self.snip_weight = snip_weight
        self.var_weight = var_weight

    def score(self, model, criterion, dataloader, device):
        mag_scores = {}
        snip_scores = {}
        var_scores = {}

        # Calculate Magnitude scores
        for _, p in self.masked_parameters:
            mag_scores[id(p)] = torch.clone(p.data).detach().abs_()

        # Debug: Print Magnitude scores
        print("Magnitude scores:", mag_scores)

        # Normalize Magnitude scores layer-wise
        for key in mag_scores:
            mag_norm = torch.sum(mag_scores[key])
            mag_scores[key].div_(mag_norm)

        # Debug: Print Normalized Magnitude scores
        print("Normalized Magnitude scores:", mag_scores)

        # Calculate Variance-based scores
        for _, p in self.masked_parameters:
            var_scores[id(p)] = torch.var(p.data).detach().abs_()

        # Debug: Print Variance-based scores
        print("Variance-based scores:", var_scores)

        # Normalize Variance-based scores layer-wise
        for key in var_scores:
            var_norm = torch.sum(var_scores[key])
            var_scores[key].div_(var_norm)

        # Debug: Print Normalized Variance-based scores
        print("Normalized Variance-based scores:", var_scores)

        # Allow masks to have gradient
        for mask, param in self.masked_parameters:
            mask.requires_grad = True

        # Ensure model is in train mode
        model.train()
        criterion.train()

        # Metric logger for debugging (Assuming utils.MetricLogger is defined elsewhere)
        metric_logger = utils.MetricLogger(delimiter="  ")

        for samples, targets in metric_logger.log_every(dataloader, 10, 'SNIP Scoring:'):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Backward pass
            model.zero_grad()
            loss.backward()

        # Calculate SNIP scores |g * theta|
        for m, p in self.masked_parameters:
            snip_scores[id(p)] = torch.clone(m.grad).detach().abs_() * torch.clone(p.data).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # Debug: Print SNIP scores
        print("SNIP scores:", snip_scores)

        # Normalize SNIP scores layer-wise
        for key in snip_scores:
            snip_norm = torch.sum(snip_scores[key])
            snip_scores[key].div_(snip_norm)

        # Debug: Print Normalized SNIP scores
        print("Normalized SNIP scores:", snip_scores)

        # Combine all normalized scores using weighted sum
        for _, p in self.masked_parameters:
            combined_score = (self.mag_weight * mag_scores[id(p)] + 
                              self.snip_weight * snip_scores[id(p)] +
                              self.var_weight * var_scores[id(p)])
            self.scores[id(p)] = combined_score

        # Debug: Print Combined scores before normalization
        print("Combined scores before normalization:", self.scores)

        # Normalize combined scores layer-wise
        for key in self.scores:
            combined_norm = torch.sum(self.scores[key])
            self.scores[key].div_(combined_norm)

        # Debug: Print Combined scores after normalization
        print("Combined scores after normalization:", self.scores)

from torch.distributions import Normal
class BayesianPruner(Pruner):
    def __init__(self, masked_parameters):
        super(BayesianPruner, self).__init__(masked_parameters)

    def score(self, model, criterion, dataloader, device):
        bayesian_scores = {}

        # Assume a prior distribution for each parameter
        prior_mu = 0.0
        prior_sigma = 1.0
        prior_distribution = Normal(prior_mu, prior_sigma)

        # Set up variational parameters
        var_mu = {}
        var_rho = {}
        for _, p in self.masked_parameters:
            var_mu[id(p)] = torch.nn.Parameter(torch.clone(p.data).detach())
            var_rho[id(p)] = torch.nn.Parameter(torch.ones_like(p.data))

        # Variational inference loop
        optimizer = torch.optim.Adam(var_mu.values(), lr=1e-3)

        for epoch in range(10):  # Adjust the number of epochs as needed
            model.train()
            criterion.train()

            for samples, targets in dataloader:
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Sample weights from the variational distribution
                for _, p in self.masked_parameters:
                    epsilon = torch.randn_like(var_rho[id(p)])
                    p.data = var_mu[id(p)] + torch.log1p(torch.exp(var_rho[id(p)])) * epsilon

                # Forward pass
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                # KL divergence between the prior and the variational posterior
                kl_divergence = 0.0
                for _, p in self.masked_parameters:
                    q_distribution = Normal(var_mu[id(p)], torch.log1p(torch.exp(var_rho[id(p)])))
                    kl_divergence += torch.sum(torch.distributions.kl.kl_divergence(q_distribution, prior_distribution))

                # Total loss is the sum of the data loss and the KL divergence
                total_loss = loss + kl_divergence

                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        # Calculate Bayesian scores as the variance of the variational posterior
        for _, p in self.masked_parameters:
            bayesian_scores[id(p)] = torch.log1p(torch.exp(var_rho[id(p)])).detach().abs_()

        # Normalize Bayesian scores
        all_bayesian_scores = torch.cat([torch.flatten(v) for v in bayesian_scores.values()])
        bayesian_norm = torch.sum(all_bayesian_scores)
        for key in bayesian_scores:
            bayesian_scores[key].div_(bayesian_norm)

        self.scores = bayesian_scores

        # Debug: Print Bayesian scores
        print("Bayesian scores:", self.scores)