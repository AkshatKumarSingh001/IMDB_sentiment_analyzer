# Simple RNN Sentiment Analysis

A practical deep-learning project that uses a Simple Recurrent Neural Network (RNN) with an embedding layer to classify IMDB movie reviews as **positive** or **negative**.

The project demonstrates the complete workflow:

- Loading and exploring the IMDB sentiment dataset
- Preparing and padding variable-length review sequences
- Learning word representations with an embedding layer
- Training and evaluating a Simple RNN model
- Predicting sentiment for new, user-provided reviews

## Project Structure

```text
Sec - 56 Simple RNN/
├── SimpleRNN/
│   ├── Simplernn.ipynb       # End-to-end dataset, training, and evaluation workflow
│   ├── embedding.ipynb       # One-hot encoding and word-embedding demonstration
│   └── prediction.ipynb      # Load the trained model and predict review sentiment
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9 or newer
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook or JupyterLab

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

## Usage

Open the notebooks from the `SimpleRNN` directory and run them in this order:

1. **`embedding.ipynb`**  
   Explore one-hot encoding, padding, and word embeddings.

2. **`Simplernn.ipynb`**  
   Load the IMDB dataset, preprocess review sequences, build and train the Simple RNN model, and evaluate its performance.

3. **`prediction.ipynb`**  
   Load the saved model and classify custom movie reviews as positive or negative.

The prediction notebook expects the trained model file to be available at:

```text
SimpleRNN/simple_rnn_model.h5
```

## Model Overview

The model uses:

1. An embedding layer to convert word indices into dense vectors
2. A Simple RNN layer to process the review sequence
3. A dense output layer with sigmoid activation for binary sentiment classification

The IMDB dataset is limited to a fixed vocabulary and padded sequence length so that reviews can be processed in batches.

## Example

```python
review = "The movie was engaging, well acted, and thoroughly enjoyable."
```

The prediction workflow returns the predicted sentiment and its confidence score.

## Notes

- The IMDB dataset may be downloaded automatically by Keras the first time it is used.
- Trained model files and local virtual environments are excluded from version control through `.gitignore`.
- Results may vary depending on the TensorFlow version, training configuration, and hardware.

## License

This project is provided for educational and demonstration purposes.