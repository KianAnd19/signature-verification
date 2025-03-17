# Signature Verification using Siamese Neural Networks

This project implements a signature verification system using Siamese Neural Networks (SNN) with PyTorch. The system is designed to compare two signature images and determine if they are from the same person.

## Features

- Siamese Neural Network architecture for signature comparison
- Convolutional Neural Network (CNN) for feature extraction
- Data loading and preprocessing using PyTorch's Dataset and DataLoader
- Training with mixed precision using torch.cuda.amp
- Learning rate scheduling with StepLR
- Validation during training
- Model saving and loading
- Test functionality for individual image pairs and bulk testing'

The model is able to accurately distinguish between forged and genuine signatures:

**Genuine:**

![image](images/genuine.png) 


**Forged:**

![image](images/forged.PNG) 

## Requirements

- Python 3.x
- PyTorch
- torchvision
- pandas
- Pillow
- tqdm
- kagglehub

## Project Structure

- `train.py`: Main script for training and testing the model
- `network.py`: Contains the SNN and CNN model definitions

## Usage

The dataset should download automatically if kagglehub is installed

### Training the Model

To train the model, run:

```
python train.py train
```

This will start the training process using the data specified in `sign_data/train_data.csv`. The best model will be saved as `best_model.pth`, and the final model will be saved as `model_last.pth`.

### Testing the Model

To test the model on the entire test set, run:

```
python train.py val 
```

This will evaluate the model on all pairs specified in `sign_data/test_data.csv` and print the overall accuracy.

### Using the Model

To use the trained model for comparing two specific signatures, you can use the `image_similarity()` function in `train.py` or call it from the command line. Example usage:

```
python train.py test 'file_path1' 'file_path2'
```

```python
result = image_similarity('path/to/image1.jpg', 'path/to/image2.jpg')
print(f"Similarity score: {result}")
```

**Note: The similarity scores are reverse (1 = Forged, 0 = Genuine)**

## Model Architecture

![image](images/architecture.png)


The model consists of two main components:

1. A Convolutional Neural Network (CNN) for feature extraction
2. A Siamese Neural Network (SNN) that uses the CNN to compare two images

The CNN architecture includes:
- 4 convolutional layers with ReLU activation and max pooling
- A fully connected layer that outputs a 128-dimensional feature vector

The SNN takes the absolute difference between the feature vectors of two images and passes it through a final fully connected layer to produce a similarity score.

## Hyperparameters

- Learning rate: 0.002
- Batch size: 64
- Epochs: 10

## Dataset

The [dataset](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) was aquired from kaggle, it is a restructured version of a dataset from a competition on detecting signature authenticity. It contains genuine and forged signatures of varying degrees, some of the forged signatures are from skilled forgers and the rest are from ordinary people trying to replicate signatures.

## Performance

The model was trained for 10 epochs, achieving 100% accuracy on the 10th epoch.

![image](images/metrics.png)

## Future Improvements

- Implement data augmentation to increase the robustness of the model
- Experiment with different CNN architectures or pre-trained models
- Add more evaluation metrics such as False Accept Rate (FAR) and False Reject Rate (FRR)
- Implement cross-validation for more reliable performance estimation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgements

- https://medium.com/analytics-vidhya/siamese-neural-network-for-signature-verification-efd2f399d586(Providing a general overview on the problem, and also the basis of the neural network architecture.)
- [dataset](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset)
