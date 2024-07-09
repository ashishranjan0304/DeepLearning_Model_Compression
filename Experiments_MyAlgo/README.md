# OurAlgo: Combined Gradient and Magnitude-Based Pruning

## Overview

OurAlgo is a network pruning method that combines gradient-based and magnitude-based pruning techniques to identify and remove less important connections in a neural network. This hybrid approach evaluates both the magnitude of the network parameters and the sensitivity of the loss function to these parameters, ensuring that only the most crucial connections are retained.

## Main Points

1. **Combined Pruning Approach**: OurAlgo integrates magnitude-based and gradient-based pruning methods to leverage the strengths of both approaches.
2. **Connection Sensitivity and Magnitude**: The importance of each connection is determined by combining the magnitude of the parameters and the gradient of the loss function with respect to these parameters.
3. **Efficiency**: By pruning the network using a combined scoring method, OurAlgo aims to reduce training time and computational resources while maintaining model performance.

## Algorithm

![OurAlgo Algorithm](https://github.com/ashishranjan0304/DeepLearning_Model_Compression/blob/master/images/our_algo.png
)

## Limitations

- **Complexity of Combination**: The combination of magnitude and gradient scores requires careful tuning of the parameters α and β to achieve optimal results.
- **Initial Sensitivity Measurement**: The initial measurement might not fully capture all important connections, especially in highly complex networks.

## Conclusion

OurAlgo is an effective pruning method that combines gradient-based and magnitude-based techniques to identify and remove less important connections in a neural network. By focusing on both connection sensitivity and magnitude, it retains essential connections and reduces computational overhead, making it suitable for various models and applications.