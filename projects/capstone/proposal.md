# Machine Learning Engineer Nanodegree
## Capstone Proposal
Pushkar Varma  
January 6, 2018

## Proposal

### Domain Background

The general idea for this project was taken from a Kaggle competition initiated by State Farm.  Car accidents are caused by many reasons, but according to the CDC, about 20% of those accidents are due to distracted drivers.  This translates to 391,000 people injured and 3,477 people killed by distracted driving, based on 2015 data by the CDC, and 2015 has had the largest number of distracted driving deaths since 2010.  The number of deaths due to distracted driving can be reduced through both social and technical means.  This project discusses how technical means can be used to detect distracted driving.  If distracted driving can be detected effectively, drivers can be alerted quickly before accidents occur.  Additionally, opportunities may arise in helping detect other kinds of impaired driving scenarios such as drunk driving, which is also a major cause of deaths on the road.  

Based on data from NHTSA, 16-24 years old have the highest cell phone use; this directly correlates to
There are various types of distractions: cognitive, visual and manual.  The manual distractions are easier to detect due to physical spatial movements that deviate from the nominal posture for driving.  " Teens were the largest age group reported as distracted at the time of fatal crashes." [3]  Based on electronic device use in the US, there has been an increasing trend in "visible manipulation of handheld devices" from 2006 to 2015. [4]

Detecting various distracted behaviors can help improve driver behavior and prevent deaths.  Additional opportunities can arise in helping insurance companies optimize their insurance policies for customers willing to integrate such technical mechanisms and share their driving behavior with insurance companies.

### Problem Statement

The problem is to detect distracted driving behaviors in camera images and classify driver behavior as being in one of a pre-defined set of behavior classes, such as normal driving, texting, and drinking, for a total of 10 different classes.
The camera images can be processed using deep learning, in particular Convolutional Neural Networks (CNN), and classification accuracy can be measured to gauge effectiveness of the model.  Based on the effectiveness of the model, in reality, the model can be deployed in camera mounted devices within cars to warn users when distracted driving behavior is detected.

### Datasets and Inputs

The input dataset will be taken from the Kaggle competition for distracted driving, as provided in reference [6].
The dataset contains 22424 training images and 79726 testing images, created by StateFarm with various distracted driver positions.  The training images are already stored in folders representing a specific class.
Each image size is 640x480 and is a color JPG file.

This dataset is being used since it is a public dataset provided by StateFarm and is a large set specifically created for covering a large class of distractions that most commonly occur.  As part of the submission of this Capstone project, a small subset will be provided for evaluation purposes.

### Solution Statement

The solution will consist of a machine learning pipeline with pre-processing, training, testing, and accuracy measurement stages.  The solution will use Convolutional Neural Networks (CNNs) since the input data is a set of images, i.e. 2-D tensors, and CNNs are best for such input.  In the pre-processing stage, the images will be pre-processed to 224x224x3 , with the same aspect ratio, in order to reduce processing time.  The images will be rescaled, and will be transformed to a grey scale for model training and prediction.  Additionally, the CNN may use pooling to allow for position invariance, softmax fox classifying based on likelihood since the output will  based on a set of mutually exclusive classes, and possibly use regularization methods such as "dropout" to gain processing efficiency and reduce overfitting.   The output will be chosen based on maximum likelihood of a class and compared with target label, and classification accuracy percentage will be calculated.

### Benchmark Model
_(approximately 1-2 paragraphs)_

The benchmark model that will be used are the results obtained in the whitepaper, [5], on the same Statefarm dataset.
The whitepaper entitled, "Realtime Distracted Driver Posture Classification", uses the same Statefarm dataset trained with genetically weighted ensemble of CNNs to obtain a classification accuracy of 95.98%.

CNNs will also be used here with a different design to obtain a classification accuracy and compared to the one in the whitepaper to determine whether the CNN design is good or needs to be improved.  Further research will be done to determine whether a more complex CNN is desirable or an ensemble is more appropriate.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

Classification accuracy will be used as a primary metric to evaluate the performance of the trained model.   The accuracy will be simply based on the ratio of the number of images classified accurately to the total number of images.  Each image will be classified accurately if the class identified by the model is the same as the label for the image.  This percentage will be used to compare to the benchmark described above.

### Project Design
_(approx. 1 page)_

One possible design is the use of a Convolution Neural Network (CNN)

![Project Design](./cnn-design-1.png)

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?

**Reference**
[1] https://www.kaggle.com/c/state-farm-distracted-driver-detection
[2] https://www.cdc.gov/motorvehiclesafety/distracted_driving/index.html
[3] https://www.nhtsa.gov/risky-driving/distracted-driving
[4] https://www.nhtsa.gov/sites/nhtsa.dot.gov/files/documents/driver_electronic_device_use_in_2015_0.pdf
[5] Realtime Distracted Driver Posture Classification - https://arxiv.org/pdf/1706.09498.pdf
[6] https://www.kaggle.com/c/state-farm-distracted-driver-detection/download/imgs.zip
[7] Benchmark of Deep Learning Models on Large Healthcare MIMIC Datasets, https://arxiv.org/pdf/1710.08531.pdf
[8] Metrics To Evaluate Machine Learning Algorithms in Python, https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/
