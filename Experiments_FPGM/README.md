# FPGM: Filter Pruning via Geometric Median

## Overview

Filter Pruning via Geometric Median (FPGM) is a novel method for compressing deep convolutional neural networks (CNNs). Unlike traditional norm-based pruning methods that remove filters with the smallest norms, FPGM prunes filters that are deemed redundant based on the geometric median. This approach ensures that the remaining filters retain the critical information necessary for maintaining model performance.

## Main Points

1. **Norm-based Pruning Limitations**: Traditional pruning methods based on filter norms have two main requirements:
   - The norm deviation of the filters should be large.
   - The minimum norm of the filters should be small.
   These requirements are not always met in practical scenarios, leading to suboptimal pruning results.

2. **Geometric Median**: The geometric median is used to identify filters that can be replaced by others with minimal impact on the model's performance. This approach is more robust and does not rely on the norm of the filters.

3. **Performance**: FPGM has been shown to achieve significant reductions in the number of floating-point operations (FLOPs) while maintaining or even improving model accuracy.

## Algorithm

<<<<<<< HEAD
### FPGM Algorithm

1. **Initialize**: 
   - Initialize the model parameters \( W \).
2. **Train**: 
   - Train the model for the specified number of epochs using the training data \( X \).
3. **Compute Geometric Median**: 
   - For each layer \( i \), compute the geometric median \( x_{GM} \) of the filters \( F_{i,j} \):
     \[
     x_{GM} = \arg \min_{x \in \mathbb{R}^{N_i \times K \times K}} \sum_{j=1}^{N_{i+1}} \|x - F_{i,j}\|_2
     \]
4. **Find Redundant Filters**: 
   - Find filters nearest to the geometric median and prune them:
     \[
     F_{i,j} = \arg \min_{F_{i,j}} \|F_{i,j} - x_{GM}\|_2
     \]
5. **Zeroize Selected Filters**: 
   - Zeroize the selected filters to prune them.
6. **Extract Pruned Model**: 
   - Extract the pruned model parameters \( W^* \).
=======
![FPGM Algorithm](/../images/FPGM_algo.png)
>>>>>>> changes

## Limitations

- **Computation Overhead**: Computing the geometric median can be computationally intensive, especially for large networks.
- **Pruning Interval**: The performance of pruning may vary depending on the interval at which pruning is performed during training.
- **Dependency on Data**: The effectiveness of pruning can be influenced by the nature of the training data and the distribution of filter norms.

<<<<<<< HEAD
## Application on Different Models

### ResNet

- **Dataset**: CIFAR-10 and ILSVRC-2012
- **Results**:
  - On CIFAR-10, FPGM reduced more than 52% FLOPs on ResNet-110 with a 2.69% relative accuracy improvement.
  - On ILSVRC-2012, FPGM reduced more than 42% FLOPs on ResNet-101 without a top-5 accuracy drop.

### VGG

- **Dataset**: CIFAR-10
- **Results**:
  - FPGM outperformed norm-based pruning methods by achieving better accuracy with a similar or higher reduction in FLOPs.

### DETR

- **Dataset**: Custom dataset
- **Results**:
  - FPGM successfully pruned DETR models, maintaining detection accuracy while reducing computational complexity.

## Visuals

### Pruning Criterion Illustration

![Pruning Criterion](path/to/pruning_criterion_image.jpg)

### Norm Distribution in Real Scenarios

![Norm Distribution](path/to/norm_distribution_image.jpg)

### Feature Map Visualization

![Feature Map Visualization](path/to/feature_map_visualization.jpg)

## Conclusion

FPGM provides an effective approach for compressing CNNs by pruning redundant filters based on the geometric median. It addresses the limitations of norm-based pruning methods and demonstrates significant improvements in both theoretical and practical performance.

=======
## Conclusion

FPGM provides an effective approach for compressing CNNs by pruning redundant filters based on the geometric median. It addresses the limitations of norm-based pruning methods and demonstrates significant improvements in both theoretical and practical performance.
>>>>>>> changes
