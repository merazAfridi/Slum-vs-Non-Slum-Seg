# Slum Detection in Mirpur, Dhaka using Satellite Imagery

## Overview
This project aims to detect slum areas in Mirpur, Dhaka, using satellite imagery. By utilizing deep learning techniques, particularly image segmentation models, we aim to identify regions within the images that correspond to slum areas. This can provide valuable insights for urban planning, resource allocation, and social development.

## Dataset
The dataset consists of satellite images covering Mirpur, Dhaka. Specifics of the dataset include:
- **33 images**: Satellite imagery of the target area.
- **18 usable images**: After filtering out non-relevant images.
- **30 masks**: Corresponding segmentation masks indicating slum regions.
- **3 corrupted JSON files**: These files were discarded during preprocessing.
- The dataset is split into training and testing sets.

## Preprocessing
To prepare the data for model training:
1. **Image Resizing**: All images were resized to **512x512 pixels**.
2. **Normalization**: Images were normalized using ImageNet statistics (mean and standard deviation).
3. **Data Augmentation**: To improve generalization, we performed various augmentations such as:
   - Random horizontal flips
   - Random rotations
   - Random adjustments of brightness and contrast
4. **Feature Enhancement**: While not explicitly coded, potential feature enhancement strategies like edge detection or texture analysis were considered.

## Model Architecture
For this task, we used a **U-Net** architecture with a **ResNet50 encoder**. This combination provides robust feature extraction capabilities, which are crucial for precise segmentation. The model is designed for **binary segmentation**, where each pixel is classified as part of a slum or not.
- **Output layer**: The final layer has a sigmoid activation function to output probabilities.

## Loss Function & Optimizer
To effectively train the segmentation model, we used a **hybrid loss function** combining **Dice Loss** and **Focal Loss**. This approach helps address challenges like class imbalance and overlapping features in the slum areas.
- **Optimizer**: We used **Adam** with a learning rate of **1e-4** for efficient gradient descent.

## Training Setup
- **Epochs**: 100
- **Batch Size**: 16 (depending on GPU availability)
- **Metrics**: During training, we tracked:
  - **Loss**
  - **Dice Coefficient**: Measures the similarity between predicted and actual masks.
  - **Jaccard Index**: Measures the overlap between predicted and true masks.
  - **Pixel Accuracy**: The percentage of correctly classified pixels.
  
**Validation**: After each epoch, we validated the model's performance on a separate validation set to monitor overfitting and adjust hyperparameters if necessary.

## Evaluation Results
Upon testing the model on the unseen data, the following results were obtained:
- **Test Loss**: 0.1596
- **Test Accuracy**: 87.74%
- **Test Dice Coefficient**: 0.5460
- **Test Jaccard Index**: 0.3784

These results demonstrate the model's ability to correctly classify slum areas in the satellite images.

## Visualizations
To better understand the model's performance, we created several visualizations:
1. **Loss and Accuracy Curves**: Plotted over the course of the training to show model convergence.
2. **Segmentation Masks**: We visualized the original image, the predicted segmentation mask, and an overlay of the predicted mask on the original image to illustrate model accuracy.
3. **Grad-CAM Heatmaps**: Used Grad-CAM to generate heatmaps that highlight the regions of the image that the model focused on when making predictions.

## Insights & Challenges
- **Segmentation Approach**: The U-Net architecture with a ResNet50 encoder worked well for detecting slum areas. The binary segmentation task was effective in distinguishing slum regions.
- **Handling Class Imbalance**: The hybrid loss function helped mitigate class imbalance by giving more weight to harder-to-classify areas.
- **Challenges**:
  - **Limited Dataset**: The dataset only had 18 usable images, which restricted the model's ability to generalize. Expanding the dataset with more diverse satellite images could help.
  - **Class Imbalance**: Slum regions may be underrepresented in the data, which was handled with the hybrid loss but still remains a challenge.
  - **Corrupted Data**: Some JSON files were corrupted and had to be excluded, affecting the data's completeness.
  - **Boundary Refinement**: The model showed issues with fine boundaries around slum areas, which could be improved with more advanced post-processing techniques.

## Conclusion
The segmentation model provides a solid foundation for slum detection in Mirpur, Dhaka, based on satellite imagery. Despite challenges such as limited data and class imbalance, the model achieved strong results, making it useful for urban planning and resource allocation. Future work can focus on improving boundary detection, expanding the dataset, and exploring alternative models.

## Tools & Libraries
- **Hardware**: Kaggle P100 GPU (for model training).
- **Language**: Python 3.x
- **Libraries**:
  - **PyTorch** (for deep learning model implementation)
  - **Albumentations** (for data augmentation)
  - **OpenCV** (for image processing)
  - **Matplotlib** (for visualizations)
  - **PIL** (for image processing)
  - **Segmentation Models PyTorch (SMP)** (for U-Net architecture)
  - **Grad-CAM** (for heatmap visualizations)

## How to Run
1. Clone the repository.
2. Install the required dependencies (refer to `requirements.txt`).
3. Download the dataset (provided in the project description).
4. Run the training script to begin training the model.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
