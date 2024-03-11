# AI ART AUTHENTICATOR
## Abstract
The AI Art Authenticator: Deep Learning Image Classification project leverages deep learning techniques to tackle the challenging task of distinguishing between Images that are produced using artificial intelligence and those that are not. With a comprehensive dataset consisting of 5000 Images that are produced using artificial intelligence and those made by humans for training, and an additional 3000 images in each category for testing, this project achieves an accuracy rate of 87.95%. A CNN architecture was developed for the project's core component utilising TensorFlow and Keras. This CNN has layers that include convolutional, pooling, and fully connected layers, and its final layer uses sigmoid activation for binary classification. When training the model, enhanced picture data is utilised to increase the model's robustness and generalizability.
## Methodology
The methodology of the project involves using deep learning techniques, specifically CNN, to classify images into two categories: AI produced and not AI produced. The project utilizes a dataset consisting of 5000 AI produced images and non-AI produced images for training and 3000 AI-generated images and 3000 non-AI-generated images for testing. The CNN model is trained using data augmentation techniques to improve its performance. The model architecture includes multiple convolutional layers followed by max-pooling layers, fully connected layers, and dropout layers to prevent overfitting. The model is trained using binary cross-entropy loss and Adam optimizer. After training, the model is used to predict the labels of test images, and the results are evaluated using a confusion matrix to assess its accuracy.
## Data Collection and Processing
The success of any machine learning project relies heavily on the quality and quantity of data available for training and evaluation. In this study, we collected a dataset consisting of 3000 AI produced images and non AI produced images for training purposes. An additional 3000 AI-generated images and 3000 non AI produced images were set aside for testing the model's performance.
All the images were downsized to the same 224x224 pixel size in order to prepare the data. We employed data augmentation methods such rotation, breadth and height shifting, shear, zooming, and horizontal flipping to broaden the diversity of the training sample. The pixel values of all photos were normalised to lie inside the [0, 1] range to help with convergence during training.
The dataset used in this project is available in [Here](https://www.kaggle.com/datasets/kausthubkannan/ai-and-human-art-classification).