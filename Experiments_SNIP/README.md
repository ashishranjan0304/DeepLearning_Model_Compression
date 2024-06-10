# SNIP: Single-shot Network Pruning based on Connection Sensitivity

## Overview

SNIP (Single-shot Network Pruning based on Connection Sensitivity) is a pruning method designed to identify and remove less important connections in a neural network before training. This method evaluates the sensitivity of the loss function to the connections in the network, retaining only the most crucial ones.

## Main Points

1. **Single-shot Pruning**: SNIP performs pruning in a single step before training, reducing computational overhead.
2. **Connection Sensitivity**: The importance of each connection is determined based on the sensitivity of the loss function, ensuring that only essential connections are retained.
3. **Efficiency**: By pruning the network before training, SNIP significantly reduces the training time and computational resources required.

## Algorithm

![SNIP Algorithm](https://github.com/ashishranjan0304/DeepLearning_Model_Compression/blob/master/images/SNIP_algo.png)

## Limitations

- **Initial Sensitivity Measurement**: The initial sensitivity measurement might not capture all important connections, especially in complex networks.
- **Pruning Granularity**: The method operates at the connection level, which may not always align with higher-level structural optimizations.

## Conclusion

SNIP is an effective pre-training pruning method that identifies and removes less important connections in a neural network. By focusing on connection sensitivity, it retains essential connections and reduces computational overhead, making it suitable for various models and applications.
