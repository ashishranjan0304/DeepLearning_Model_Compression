# Deep Learning Model Compression Interface

## Description
This project provides a user-friendly web interface for compressing deep learning models. It supports various pruning algorithms and allows for easy configuration of model training parameters.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Screenshots](#screenshots)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/deep-learning-model-compression.git
    ```
2. Navigate to the project directory:
    ```bash
    cd deep-learning-model-compression
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    python app.py
    ```
5. Open your browser and go to `http://localhost:5000`.

## Usage

1. **Select Model:**
    - Choose the model you want to compress from the dropdown menu.
    - ![Model Selection](path_to_image/model_selection.png)

2. **Select Algorithm:**
    - Based on the selected model, choose an appropriate pruning algorithm.
    - ![Algorithm Selection](path_to_image/algorithm_selection.png)

3. **Configure Parameters:**
    - Depending on the selected algorithm, fill in the required training parameters.
    - ![Configure Parameters](path_to_image/configure_parameters.png)

4. **Run Command:**
    - Click the "Run Command" button to start the training and pruning process.
    - The output will be displayed in real-time in the output section.
    - ![Run Command](path_to_image/run_command.png)

5. **View Training Accuracy:**
    - A graph showing the training accuracy will be updated periodically.
    - ![Training Accuracy](path_to_image/training_accuracy.png)

## Features

- **Model Selection:** Choose from different models available for compression.
- **Algorithm Selection:** Various pruning algorithms like FPGM, L1, L2, SNIP, Synflow, and your custom algorithm `our_algo`.
- **Parameter Configuration:** Input fields for configuring model training parameters.
- **Real-time Output:** Monitor the training process in real-time.
- **Training Accuracy Visualization:** View the training accuracy graph which updates periodically.

## Screenshots

### Model Selection
![Model Selection](path_to_image/model_selection.png)

### Algorithm Selection
![Algorithm Selection](path_to_image/algorithm_selection.png)

### Configure Parameters
![Configure Parameters](path_to_image/configure_parameters.png)

### Run Command
![Run Command](path_to_image/run_command.png)

### Training Accuracy
![Training Accuracy](path_to_image/training_accuracy.png)



