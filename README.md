# Jellyfish Classification with CNN (PyTorch)

## Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images of jellyfish into one of six species:

- Moon
- Barrel
- Blue
- Compass
- Lion's Mane
- Mauve

The model consists of:

- Two convolutional layers with max pooling and ReLU activation
- Three fully connected layers with ReLU activation

The project also includes a Flask API for image classification.

## Dataset

The model is trained on the [Jellyfish Types dataset](https://www.kaggle.com/datasets/anshtanwar/jellyfish-types). To use this dataset:

1. Download the dataset from the link above.
2. Extract it into the `data/` directory.

## Installation

It is recommended to use a Python virtual environment for this project.

### Setup

```sh
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install torch torchvision numpy matplotlib flask
```

## Training the Model

To train the model, run:

```sh
python -m training.train
```

This will train the CNN using the dataset and save the trained model for later evaluation and inference.

## Evaluating and Using the Model

Once trained, the model can be used for evaluation and inference through dedicated modules.

## Running the Flask API

The project includes a Flask API to classify jellyfish images. To start the API, run:

```sh
python -m app.api
```

The API provides an endpoint:

- `POST /api` — Accepts a .jpg image of a jellyfish and returns a JSON response in the format:
  ```json
  { "classification": "species_name" }
  ```

### Example

```sh
curl.exe -X POST -F "image=@path/to/image.jpg" http://127.0.0.1:5000/api
```

## Project Structure

```
jellyfish/
├── app/
│   ├── __init__.py
│   ├── api.py          # Flask API for classification
│   ├── inference.py    # Inference script for classification
├── data/
│   ├── Train_Test_Valid/
│   │   ├── Train/      # Training dataset
│   │   └── test/       # Test dataset
│   ├── data_loader.py  # Data loader script
├── models/
│   ├── __init__.py
│   ├── model.py        # Model definition
│   └── model.pth       # Trained model storage
├── training/
│   ├── __init__.py
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation script
├── utils/
│   ├── __init__.py
│   ├── logger.py       # Logging utility
│   ├── visualisations.py # Visualization utility
├── venv/               # Virtual environment directory
├── .gitignore          # Git ignore file
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## License

This project is open-source. Feel free to modify and use it as needed.

---

### Contributions & Issues

Feel free to submit a pull request or report any issues if you have suggestions or encounter problems!
