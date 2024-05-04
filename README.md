# Gender Classification from CCTV Imagery

## Overview
This project aims to develop a robust machine learning model capable of classifying gender from images captured by CCTV cameras. Using a dataset of images labeled as 'MALE' or 'FEMALE', this project employs several deep learning models to train and evaluate performance, combining advanced image processing techniques with state-of-the-art convolutional neural networks (CNNs) to achieve high classification accuracy.

## Data Preprocessing
The raw images are first organized into a structured DataFrame, categorizing them by their labels and associated file paths. The dataset is then split into training, validation, and test sets to ensure a diverse representation of data across each phase of model training and evaluation.

## Model Development
Several models are explored and trained in this project:
1. **Custom CNN**: A custom-built convolutional neural network with multiple layers designed to extract and learn from the features of the input images.
2. **MobileNetV2**: Utilizing the MobileNetV2 architecture, known for its efficiency in mobile and edge devices, adapted for gender classification by modifying the top layers to suit the binary classification task.
3. **ResNet152**: Leveraging the deep residual learning framework to enable the training of deeper networks, with the ResNet152 model pretrained on ImageNet and fine-tuned for the gender classification task.
4. **ConvNeXtXLarge**: Exploring a more recent and advanced CNN architecture that offers improved performance and scalability.

Each model is fine-tuned on the training data using specific data augmentation techniques and optimized for better performance via transfer learning where applicable.

## Evaluation and Optimization
The models are evaluated using the test dataset to compare their performance based on accuracy metrics. Techniques like saliency mapping and Grad-CAM are employed to visualize model decisions, providing insights into which parts of the images are most informative for predicting gender. These visualizations help in understanding the model's focus and can guide further refinement.

## Conclusion
This project demonstrates the application of deep learning to a practical problem in surveillance and security domains. Through rigorous training, evaluation, and optimization, the developed models aim to assist in automating the monitoring process, enhancing both efficiency and reliability.

This comprehensive analysis and application of machine learning techniques provide a foundation for further research and development in automated image-based classification systems.
