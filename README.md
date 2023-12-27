# Football Club Logo Recognizer

## Overview

This repository contains the code and documentation for the "Football Club Logo Recognizer" project. The goal of this project is to recognize and classify images of football club logos from the top 5 football leagues using pattern recognition and machine learning techniques.

## Important Links
- [Web Application]()
- [Dataset (Kaggle)](https://www.kaggle.com/datasets/alexteboul/top-5-football-leagues-club-logos)


## Project Directory Structure

| File                                   | Description                              |
|----------------------------------------|------------------------------------------|
| model                                  | Folder for trained machine learning models|
| notebook_football_club_recognizer.ipynb | Jupyter notebook with the project code   |
| report_football_club_recognizer.pdf  | Detailed project report                  |
| requirements.txt                       | Project dependencies                     |
| webapp.py                              | Streamlit web application script            |

### `assets` directory
| File                       | Description                |
|----------------------------|----------------------------|
| class_mapping.json         | Mappings of clubs with corresponding class labels    |
| frontend_config.py          | Frontend configurations    |

### `project_data_files` directory
| File                          | Description                                           |
|-------------------------------|-------------------------------------------------------|
| best_ann_model.pth            | Saved best Artificial Neural Network model            |
| dataset_paths.csv             | Paths to the generated dataset                        |
| generated_dataset             | Folder containing the augmented images               |
| test-images                   | Folder containing test images                         |
| top-5-football-leagues        | Original dataset                                      |

### `ui` directory
| File                | Description                   |
|---------------------|-------------------------------|
| background.jpg      | Background image for the user interface   |
| style.css           | Stylesheet for the user interface          |

## Libraries Used

- `os`: File handling
- `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`: Data handling, mathematical operations, and visualization
- `cv2` (OpenCV-Python): Image operations
- `Pillow`: Image file handling
- `rembg`: Background removal
- `Tensorflow`, `Keras`: Convolutional Neural Network and Image Augmentation
- `PyTorch`: Artificial Neural Network (MLP)
- `Scikit-Learn`: Classification models and other machine learning operations
- `streamlit`: Model deployment

## Dataset Generation

The dataset, comprising 100 club logos (20 from each of the 5 leagues), underwent image augmentation using Keras' ImageDataGenerator module. The resulting dataset included 75 augmented images for each team, with a size of 64x64 pixels.

## Model Training and Classification Techniques

1. **Artificial Neural Network (ANN)**
   - Implemented using PyTorch
   - 4-layered Multi-Layer Perceptron
   - Optimizer: Adam, Loss: Cross Entropy Loss

   | Metric                      | RGB Images | Grayscale Data |
   |-----------------------------|------------|-----------------|
   | Test Accuracy               | 94.74%     | 40.64%          |


2. **Convolutional Neural Network (CNN)**
   Implemented using TensorFlow

   | Metric                      | RGB Images | Grayscale Data |
   |-----------------------------|------------|-----------------|
   | Test Accuracy               | 98.07%     | 87.7%           |

3. **Random Forest Classifier**
   Grid-search for optimal hyperparameters: { max_depth=20, n_estimators=200 }

   | Metric                      | RGB Data | Grayscale Data |
   |-----------------------------|----------|-----------------|
   | Test Accuracy               | 80.57%   | 67.79%          |

4. **KNearestNeighbors (KNN) Classifier**
   Grid-search for optimal hyperparameters: { n_neighbors=3, weights="distance", p=1 }

   | Metric                      | RGB Images |
   |-----------------------------|------------|
   | Test Accuracy               | 72.25%     |

5. **Support Vector Machine (SVM) Classifier**
   Grid-search for optimal hyperparameters: { kernel="rbf", gamma="auto", C=10 }

   | Metric                      | RGB Data | Grayscale Data |
   |-----------------------------|----------|-----------------|
   | Test Accuracy               | 90.2%    | 63.33%          |

6. **Other Models (Failed Attempt)**
   KMeans Algorithm: Unsuccessful due to dataset rotations leading to uncertain cluster centroids

## Best Model and Final Pipeline

The CNN Classifier emerged as the best model with a test accuracy of 98.07%. A pipeline was implemented for image preprocessing and model prediction, facilitating model deployment.


## Authors
- [Ashudeep Dubey](mailto:dubey.6@iitj.ac.in) (B.Tech. Electrical Engineering)
- [Tanish Pagaria](mailto:pagaria.2@iitj.ac.in) (B.Tech. Artificial Intelligence & Data Science)
- [Vinay Vaishnav](mailto:vaishnav.3@iitj.ac.in) (B.Tech. Electrical Engineering)  

(IIT Jodhpur Undergraduates)
