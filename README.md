# HYU-AI-G08  
**Hanyang University AI & Application Course Project (G08, Fall 2024)**  

## Group Members:
1. **Jan Rudkowski**, Warsaw University of Technology, jan.rudkowski.stud@pw.edu.pl
2. **Konrad Wojda**, Warsaw University of Technology, konrad.wojda.stud@pw.edu.pl
3. **Ruke Sam-Ogaga**, Rose-Hulman Institute of Technology, samogaoe@rose-hulman.edu  
4. **Isabel Pfeiffer**, University Pforzheim, Isabel.pfeiffer01@gmail.com

## Project Title: 
**Deep Fake Image Detector**

## Proposal:
In recent years, with the rapid advancement of AI and the ease of generating fake images, “deep fakes” have emerged as a significant threat. These digitally altered images, videos, and audios impact not only celebrities but also everyday individuals. This issue has gained particular attention in Korea, where deep fake crimes have reached unprecedented levels. The number of underage victims has increased 4.5 times, from 64 in 2022 to 288 over the past two years, while the total number of victims grew 3.7 times, rising from 212 to 781 during that same period according to *[Yonhap News Agency](https://en.yna.co.kr/view/AEN20240828003100315)*. This alarming growth inspired us to focus on developing a solution to detect deep fake images in hopes to help combat this growing concern.

Our objective is to build an accurate deep fake image detector by training and comparing the performance of various models. To accomplish this, we plan to utilize this *[Kaggle dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)*, which contains approximately 190,000 facial images. We will split the dataset into two parts: 80% for training and 20% for validation. The models we intend to use include EfficientNet and ResNet50, among others. Once trained, the models will be analyzed thoroughly to evaluate their effectiveness. Depending on the results, we may further extend the scope of our project.  

For training, we plan to use our local hardware. However, if additional resources are required, we will leverage AWS SageMaker within the limits of available free credits.  

You can follow the progress of our project on our *[Github](https://github.com/konradwojda/HYU-AI-G08)*
