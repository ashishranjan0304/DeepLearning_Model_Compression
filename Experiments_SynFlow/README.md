# SynFlow: Iterative Synaptic Flow Pruning

## Overview

SynFlow (Iterative Synaptic Flow Pruning) is a method designed to iteratively prune connections in a neural network while preserving the synaptic flow. This approach ensures that the pruned network remains connected and functional.

## Main Points

1. **Synaptic Flow Preservation**: SynFlow aims to preserve the synaptic flow, ensuring the pruned network remains functional.
2. **Iterative Pruning**: The method iteratively prunes connections, gradually reducing the network size while maintaining performance.
3. **Robustness**: SynFlow is robust and can be applied to various neural network architectures and applications.

## Algorithm

![SynFlow Algorithm](https://github.com/ashishranjan0304/DeepLearning_Model_Compression/blob/master/images/synflow_algo.png)

## Limitations

- **Iterative Process**: The iterative pruning process can be time-consuming, especially for large networks.
- **Computation Overhead**: Computing the synaptic flow for each iteration can be computationally intensive.


## Conclusion

SynFlow provides a robust and effective approach for pruning neural networks by preserving synaptic flow. Its iterative pruning process ensures the pruned network remains connected and functional, making it suitable for various models and applications.
