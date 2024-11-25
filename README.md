# ncRNA Classification with Machine Learning

## Project Overview

This project explores the classification of non-coding RNA (ncRNA) sequences using various machine learning models. The goal was not to achieve the best possible accuracy or efficiency but to familiarize with machine learning concepts, specifically focusing on the classification of RNA sequence data using models such as Random Forest, simple neural networks, transformers, and an LSTM-based hybrid model.

## Introduction

This project aims to classify different classes of ncRNAs based on their nucleotide sequence using machine learning models. The models tested include traditional methods like Random Forest, more complex approaches like transformers, and hybrid models that combine k-mer features with sequence data (such as LSTM-based architectures). The focus of the project was to understand how these models perform on RNA data and experiment with different methods of feature engineering and model architecture.

## Data

The dataset used for this project includes various classes of non-coding RNA sequences, such as:

- lncRNA (long non-coding RNA)
- snRNA (small nuclear RNA)
- misc_RNA (miscellaneous RNA)
- miRNA (microRNA)
- snoRNA (small nucleolar RNA)

The sequences were encoded into numerical arrays, and additional features, such as k-mers and secondary structure data, were also included to enhance classification.

### Preprocessing

- For the **LSTM model only**, sequences were encoded as integers (1-4) corresponding to nucleotides (A, C, G, T).
- Sequences longer than 1,000 nucleotides were truncated in the LSTM model, as this was the maximum sequence length that the available hardware could handle efficiently.
- K-mer features were generated to capture localized patterns in the sequence data for other models.

## Models

### Random Forest

Random Forest is an ensemble learning method based on decision trees. It was used as a baseline model to compare with other more complex models. Random Forest is effective for classification tasks due to its robustness and ability to handle large feature spaces.

### Neural Network

A simple feedforward neural network was implemented to understand how a basic neural network might handle RNA sequence classification. The model consisted of several fully connected layers and was tested using only the encoded sequence data.

### Transformer

A transformer-based model was tested to capture long-range dependencies in RNA sequences. The transformer was particularly useful for handling sequence data, where the attention mechanism allows the model to focus on important parts of the sequence.

### Hybrid LSTM

The Hybrid LSTM model combined k-mer features and RNA sequence data. It consists of:
- A dense layer for k-mer/numerical feature processing.
- An LSTM layer for handling sequence data with bidirectional layers.
- A final classification layer to predict gene types.

This hybrid approach aimed to integrate both sequence information and sequence motifs for better classification.

## Training and Evaluation

- **Data Splitting**: The dataset was split into training and testing sets.
- **Training**: Each model was trained on the training set, and hyperparameters were adjusted based on performance.
- **Evaluation**: The models were evaluated using standard classification metrics like accuracy, precision, recall, and F1-score.
- **Challenges**: Some classes, especially snoRNA, were more difficult to classify due to their unique secondary structures and high similarity to other RNA types.

## Conclusion

This project aimed to explore how different machine learning models handle ncRNA sequence data. The focus was on understanding the models' ability to classify RNA types, rather than achieving the highest accuracy. In general, models performed well on most classes, except for snoRNA, which posed a significant challenge. This suggests that more specialized features or a larger dataset might be needed to improve the classification of certain RNA types.

While the models were able to capture some useful information from the sequences, there's still room for improvement, particularly for classes like snoRNAs. Future work could involve incorporating additional structural features or using more complex models.

## Scripts and Notebooks

All scripts and Jupyter notebooks used in this project are available in the [scripts](./scripts) and [notebooks](./notebooks) directories. These contain the full implementation of the models, training processes, and evaluation methods.

## Setup Instructions

To get the project up and running on your local machine, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ncRNA-classification.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd ncRNA-classification
    ```

3. **Install the required dependencies:**
    - For non-PyTorch models:
    ```bash
    pip install -r requirements_for_torch_models.txt
    ```
    - For PyTorch-based models:
    ```bash
    pip install -r requirements_pytorch.txt
    ```

4. **Launch Jupyter Notebook to open and run the notebooks:**
    ```bash
    jupyter notebook
    ```

5. **To run the Python scripts**, go to the `scripts` directory and execute the scripts:
    ```bash
    cd scripts
    python import_and_clean.py   # For data preprocessing
    python Random_Forest.py      # For Random Forest model
    python neural_network.py     # For simple neural network model
    python transformer.py        # For transformer model
    python LSTM.py               # For LSTM model
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to fork, clone, or contribute to this repository. If you have any questions or feedback, please open an issue or submit a pull request. Thanks for exploring this project!
