# L1 Structured Filter Pruning

## Overview

L1 Structured Filter Pruning is a method that removes filters with the smallest L1 norm, assuming these filters have the least impact on the model's performance. This approach is simple yet effective in reducing the size and computational requirements of neural networks.

## Main Points

1. **L1 Norm-Based Pruning**: Filters with the smallest L1 norm are considered less important and are pruned.
2. **Structured Pruning**: Entire filters are removed, making the pruning process more efficient in reducing computational load.
3. **Simplicity and Efficiency**: The method is straightforward and easy to implement, making it a popular choice for model compression.

## Algorithm

![L1 Pruning Algorithm](https://github.com/ashishranjan0304/DeepLearning_Model_Compression/blob/master/images/L1_algo.png)

## Limitations

- **Norm Dependence**: The method assumes that filters with smaller L1 norms are less important, which may not always be true.
- **Fixed Pruning Criterion**: The pruning criterion is fixed and does not adapt to the training process or data characteristics.

## Conclusion

L1 Structured Filter Pruning is a simple and effective method for reducing the size and computational requirements of neural networks. By removing filters with the smallest L1 norm, it achieves significant model compression with minimal performance impact.
