# Breast Cancer Tumor Classification Machine Learning Model

## Project Overview

This project involves the classification of breast cancer diagnoses using the Wisconsin Breast Cancer Dataset. The goal is to develop a machine learning model that can accurately classify tumors as malignant or benign based on various features.

## Features

- **Dataset**: Wisconsin Breast Cancer Dataset (WBCD)
- **Features**: Includes attributes like radius, texture, perimeter, area, smoothness, compactness, concavity, and others.
- **Target**: Malignant or Benign

## Tech Stack

- **Programming Language**: Python
- **Libraries and Frameworks**:
  - **Pandas**: For data manipulation and analysis
  - **NumPy**: For numerical operations
  - **Scikit-learn**: For implementing machine learning algorithms
  - **Matplotlib**: For data visualization
  - **Seaborn**: For statistical data visualization
  - **Jupyter Notebook**: For interactive development and documentation

## Installation

To get started, clone the repository and install the necessary dependencies. You can do this by running:

```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Run `preprocessing.py` to clean and prepare the dataset.
2. **Model Training**: Execute `train_model.py` to train the machine learning model.
3. **Evaluation**: Use `evaluate_model.py` to evaluate the performance of the trained model.

Example command to train the model:

```bash
python train_model.py
```

## Results

- **Model Accuracy**: 97.5%
- **Confusion Matrix**:
  - True Positives (TP): 152
  - True Negatives (TN): 123
  - False Positives (FP): 5
  - False Negatives (FN): 10

**Example of Confusion Matrix**:

```
[[123   5]
 [ 10 152]]
```

## Contributing

Feel free to contribute to this project by submitting a pull request or opening an issue. Please make sure to follow the coding guidelines and ensure that your contributions are well-documented.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

