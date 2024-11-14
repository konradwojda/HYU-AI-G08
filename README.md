# HYU-AI-G08  
**Hanyang University AI & Application Course Project (G08, Fall 2024)**  

## Group Members:
1. **Jan Rudkowski**, Warsaw University of Technology, jan.rudkowski.stud@pw.edu.pl
2. **Konrad Wojda**, Warsaw University of Technology, konrad.wojda.stud@pw.edu.pl
3. **Ruke Sam-Ogaga**, Rose-Hulman Institute of Technology, samogaoe@rose-hulman.edu  
4. **Isabel Pfeiffer**, University Pforzheim, Isabel.pfeiffer01@gmail.com

## Project Title: 
**Deep Fake Image Detector**

## Introduction - Initial Proposal (Assignment 1):
In recent years, with the rapid advancement of AI and the ease of generating fake images, “deep fakes” have emerged as a significant threat. These digitally altered images, videos, and audios impact not only celebrities but also everyday individuals. This issue has gained particular attention in Korea, where deep fake crimes have reached unprecedented levels. The number of underage victims has increased 4.5 times, from 64 in 2022 to 288 over the past two years, while the total number of victims grew 3.7 times, rising from 212 to 781 during that same period according to *[Yonhap News Agency](https://en.yna.co.kr/view/AEN20240828003100315)*. This alarming growth inspired us to focus on developing a solution to detect deep fake images in hopes to help combat this growing concern.

Our objective is to build an accurate deep fake image detector by training and comparing the performance of various models. To accomplish this, we plan to utilize this *[Kaggle dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)*, which contains approximately 190,000 facial images. We will split the dataset into two parts: 80% for training and 20% for validation. The models we intend to use include EfficientNet and ResNet50, among others. Once trained, the models will be analyzed thoroughly to evaluate their effectiveness. Depending on the results, we may further extend the scope of our project.  

For training, we plan to use our local hardware. However, if additional resources are required, we will leverage AWS SageMaker within the limits of available free credits.  

You can follow the progress of our project on our *[Github](https://github.com/konradwojda/HYU-AI-G08)*

## Dataset
For this project, we chose to use a Kaggle dataset created by Manjil Karki "*[deepfake and real images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)*". The dataset includes both manipulated (fake) and real images, originally sourced from the *[OpenForensics Dataset](https://sites.google.com/view/ltnghia/research/openforensics)*. The dataset was processed to optimize outcomes for analysis. Each image is a 256 x 256 JPG-format image of a human face, labeled as either real or fake. The dataset contains approximately 190,000 images, divided into three main directories: Train (70,000 fake, 70,000 real), Test (5,492 fake, 5,413 real), and Validation (19,600 fake, 19,800 real).

## Methodology

### Used CNNs
We decided to train our models using __EfficientNet__ and __ResNet50__ convolutional neural networks and then compare their performances to determine which one demonstrates better performance for our task. Both 
EfficientNet and ResNet50 are known for their powerful feature extraction capabilities and efficiency, which is why we chose them.<br>
EfficientNet is optimized for high accuracy with fewer parameters by scaling depth, width, and resolution, making it efficient for complex tasks on limited resources.<br>
ResNet50 introduces residual connections that help prevent vanishing gradients, enabling deeper networks to learn more detailed features, which is crucial for distinguishing real from fake images.

### Infrastructure
For training the models, we decided to use __Amazon SageMaker Studio Lab__, which offers four hours of free GPU usage (NVIDIA Tesla T4). This was sufficient for our project and provided more computational power than the hardware we own. SageMaker Studio Lab offers many tools for developing and training models.

### Brief Code Explanation
In this section, the code will be described. For detailed explanations of all parts of the code, please visit *[Code Explanation](https://github.com/konradwojda/HYU-AI-G08/blob/main/docs/code_explanation.md)*. The main files of our project and their tasks are:

`dataset.py` – The script defines the `DeepFakeDataset` class, a custom PyTorch dataset for loading deepfake images. It takes a root directory containing "Real" and "Fake" subdirectories and generates a list of image file paths with corresponding labels. The class supports optional image transformations and provides methods to retrieve the dataset length and individual image-label pairs, making it suitable for loading data in model training and evaluation.

`train.py` – The script is designed for training a deepfake detection model using PyTorch. It loads and preprocesses a dataset, initializes a specified model (ResNet50 or EfficientNet), and trains it over a specified number of epochs, optimizing with cross-entropy loss and an Adam optimizer. During each epoch, the script computes accuracy, precision, recall, and F1-score metrics for model evaluation, and logs these metrics to a CSV file. The trained model state is saved to a file.

`predict.py` – This script loads a pre-trained model to classify an image as either "Real" or "Fake." It preprocesses the input image and uses a specified or default model file to load weights before making a prediction.

`eval.py` - The script evaluates and visualizes the performance of a deepfake detection model. It loads a trained model, applies it to a test dataset, and generates predictions. Using these predictions, the script computes a confusion matrix, which is then visualized and saved as an image. This matrix offers insight into the model’s performance by showing counts of correct and incorrect predictions for each class (Real vs. Fake). The script scans for model files in a specified folder, allowing for batch evaluation of multiple trained models.

`train_models.sh` – This is simple bash script used to start training on virtual machine. The script automates downloading, unzipping, and training models on a dataset. After unzipping, it trains models at different epoch counts, saving each model and optionally logging metrics.

## Evaluation & Analysis
For both models a *confussion matrix* and a metrics such as *training_loss*, *accuracy*, *precision	recall*, *f1score* have been calculated and will be shown below based on the number of training epochs.

### EfficientNet 

### ResNet50


## Related Work
## Conclusion

