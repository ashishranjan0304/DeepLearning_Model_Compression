# Deep Learning Model Compression Interface

## Description
This project provides a user-friendly web interface for compressing deep learning models. It supports various pruning algorithms and allows for easy configuration of model training parameters.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Screenshots](#screenshots)
-

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ashishranjan0304/DeepLearning_Model_Compression.git
    ```
2. Navigate to the project directory:
    ```bash
    cd DeepLearning_Model_Compression
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Install dependencies:
    ```bash
    cd user_interface
    ```
6. Install dependencies:
    ```bash
    python setup_script.py
    ```
7. Run the application:
    ```bash
    python app.py
    ```
8. Open your browser and go to `http://localhost:5000`.

## Usage

1. **Select Model:**
    - Choose the model you want to compress from the dropdown menu.
    - ![Model Selection](/static/images/select_model.png)

2. **Select Algorithm:**
    - Based on the selected model, choose an appropriate pruning algorithm.
    - ![Algorithm Selection](/static/images/select_algo.png)

3. **Configure Parameters:**
    - Depending on the selected algorithm, fill in the required training parameters.
    - ![Configure Parameters](/static/images/select_arguments.png)

4. **Run Command:**
    - Click the "Run Command" button to start the training and pruning process.
    - The output will be displayed in real-time in the output section.
    - ![Run Command](/static/images/run_cmd.png)

5. **View Training Accuracy:**
    - A graph showing the training accuracy will be updated periodically.
    - ![Training Accuracy](/static/images/output.png)

## Features

- **Model Selection:** Choose from different models available for compression.
- **Algorithm Selection:** Various pruning algorithms like FPGM, L1, L2, SNIP, Synflow, and your custom algorithm `our_algo`.
- **Parameter Configuration:** Input fields for configuring model training parameters.
- **Real-time Output:** Monitor the training process in real-time.
- **Training Accuracy Visualization:** View the training accuracy graph which updates periodically.

