# L2 Structured Filter Pruning

## Overview

L2 Structured Filter Pruning is a method that removes filters with the smallest L2 norm, assuming these filters have the least impact on the model's performance. This approach leverages the L2 norm to identify and prune less important filters.

## Main Points

1. **L2 Norm-Based Pruning**: Filters with the smallest L2 norm are considered less important and are pruned.
2. **Structured Pruning**: Entire filters are removed, making the pruning process more efficient in reducing computational load.
3. **Effectiveness**: The method is effective in reducing model size and computational requirements while maintaining performance.

## Algorithm

![L2 Pruning Algorithm](/../images/L2_algo.png)

## Limitations

- **Norm Dependence**: The method assumes that filters with smaller L2 norms are less important, which may not always be true.
- **Fixed Pruning Criterion**: The pruning criterion is fixed and does not adapt to the training process or data characteristics.

## Conclusion

L2 Structured Filter Pruning is an effective method for reducing the size and computational requirements of neural networks. By removing filters with the smallest L2 norm, it achieves significant model compression with minimal performance impact.
