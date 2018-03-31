# Machine Learning Engineer Nanodegree
## Capstone Project
Pushkar Varma
March 30, 2018

## I. Definition
_(approx. 1-2 pages)_

### Project Overview
The general idea for this project was taken from a Kaggle competition initiated by State Farm.  Car accidents are caused by many reasons, but according to the CDC, about 20% of those accidents are due to distracted drivers.  This translates to 391,000 people injured and 3,477 people killed by distracted driving, based on 2015 data by the CDC, and 2015 has had the largest number of distracted driving deaths since 2010.  The number of deaths due to distracted driving can be reduced through both social and technical means.  This project discusses how technical means can be used to detect distracted driving.  If distracted driving can be detected effectively, drivers can be alerted quickly before accidents occur.  Additionally, opportunities may arise in helping detect other kinds of impaired driving scenarios such as drunk driving, which is also a major cause of deaths on the road.  

Based on data from NHTSA, 16-24 years old have the highest cell phone use. There are various types of distractions: cognitive, visual and manual.  The manual distractions are easier to detect due to physical spatial movements that deviate from the nominal posture for driving.  " Teens were the largest age group reported as distracted at the time of fatal crashes." [3]  Based on electronic device use in the US, there has been an increasing trend in "visible manipulation of handheld devices" from 2006 to 2015. [4]

Detecting various distracted behaviors can help improve driver behavior and prevent deaths.  Additional opportunities can arise in helping insurance companies optimize their insurance policies for customers willing to integrate such technical mechanisms and share their driving behavior with insurance companies.

##### Datasets and Inputs

The input dataset will be taken from the Kaggle competition for distracted driving, as provided in reference [6].  The dataset contains 22424 training images and 79726 testing images, created by StateFarm with various distracted driver positions.  The training images are already stored in folders representing a specific class.  Each image size is 640x480 and is a color JPG file.  There are a total of 10 classes for which training images are provided and a large set of unlabeled test images is also provided.

The 10 classes are as follows with the number of training images provided for each class:

```
  c0: safe driving  (2489 images)
  c1: texting - right  (2267 images)
  c2: talking on the phone - right (2317 images)
  c3: texting - left (2346 images)
  c4: talking on the phone - left (2326 images)
  c5: operating the radio (2312 images)
  c6: drinking (2325 images)
  c7: reaching behind (2002 images)
  c8: hair and makeup (1911 images)
  c9: talking to passenger (2129 images)
```

***Figure 1- Image Classifications***

### Problem Statement
The problem is to detect distracted driving behavior postures in camera images and classify driver behavior as being in one of a pre-defined set of behavior classes, such as normal driving, texting, and drinking, for a total of 10 different classes, as described in the previous section.

The camera images will be loaded and processed using deep learning, in particular Convolutional Neural Networks (CNN), and classification accuracy will be measured to gauge effectiveness of the model.  Based on the effectiveness of the model, in reality, the model can be deployed in camera mounted devices within cars to warn users when distracted driving behavior is detected.

The classification will be performed using CNNs, with regularization techniques such as Dropout or L1 regularization to prevent overfitting, using various hyperparameter values to determine which decay values work best.  The original set of images will be divided into a training set, validation set and testing set to prevent bias and effectively measure model performance.  

A baseline model will be used to assess the performance of a basic model for distraction classification, and then several other model designs will be used to improve on classification accuracy and compare that with the baseline model. Training is done on both raw images and grayscaled/histogram equalized images to determine which results in better model performance.

The final trained model will be used to classify any given input image from the test set or the unlabeled image set in order to determine whether a image contains a distracted driver, `c0`, or a specific class of distraction, `c1` to `c9`.

### Metrics
The `accuracy` metric was used to measure how well the model was trained by evaluating the model with a test image set.  This `accuracy` metric is most appropriate since images are being classified into one of 10 classes, and a performance binary decision is made as to whether the classification is correct or not based on the predicted class vs. actual class.   For example, for a set of 100 images, if 90 of the images are classified with a predicted class equal to actual class, then the classification accuracy will be considered to be 90%.

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
There are 2 datasets provided - one is a set of labeled training images, and another set is a set of unlabeled images.  The labeled image set is split into a training, validation and testing set.  The unlabeled set is manually sampled to provide evaluations of the final trained model selected with the high evaluation accuracy.

Overall the training dataset seems balanced, other than the `c7 & c8` classes that seem to have the least number of images.  This may lead to bit more bias towards class `c0` having the most number of samples, hence classification accuracy for `c0` may be higher, and similarly for some of the other classes such as `c2-c6`.  The diagram shows a visual of the distribution of the count by class.

In order to alleviate in imbalances, the training dataset will be trimmed to ensure ***equal number of images*** exist for all classes.

![Count by Class](C:\Users\pushkar\ML\machine-learning\projects\capstone\data-count-by-class.png)

***Figure 2 - Distribution of Images by Class***

This dataset is being used since it is a public dataset provided by StateFarm and is a large set specifically created for covering a large class of distractions that most commonly occur.  As part of the submission of this Capstone project, a small subset will be provided as samples.

For this project, an equal number of images were selected for all classes in order to remove any bias during training.   A total of **1900** images were selected for each class, .e.g `c0` to `c9`.  

The following tables shows samples of an image in each class, `c0` to `c9` , and the larger sample of images has been provided in the `sample_images` folder.

|           c0![mg_3](.\sample_images\c0\img_34.jpg)           |            c1![mg_](.\sample_images\c1\img_6.jpg)            |          c2![mg_18](.\sample_images\c2\img_186.jpg)          |            c3![mg_](.\sample_images\c3\img_5.jpg)            | c4![mg_1](C:\Users\pushkar\ML\machine-learning\projects\capstone\sample_images\c4\img_14.jpg) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **c5**![mg_5](C:\Users\pushkar\ML\machine-learning\projects\capstone\sample_images\c5\img_56.jpg) | **c6**![mg_](C:\Users\pushkar\ML\machine-learning\projects\capstone\sample_images\c6\img_0.jpg) | **c7**![mg_8](C:\Users\pushkar\ML\machine-learning\projects\capstone\sample_images\c7\img_81.jpg) | **c8**![mg_2](C:\Users\pushkar\ML\machine-learning\projects\capstone\sample_images\c8\img_26.jpg) |         **c9**![mg_1](.\sample_images\c9\img_19.jpg)         |

***Figure 3 - Sample images from each class***

### Exploratory Visualization
The image set provided contains colored images of various driver postures that are in 10 different classes.  An initial look at the grayscale image and its histogram shows there are many pixels with low intensity below 120, and a smaller number of pixels with high intensity above 230.  A sample and its histogram is shown below.   The sample image's histogram is equalized and both the processed image and its new histogram is shown as well.  

| Image                                                        | Processing Results                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Original Image**![riginal-imag](.\pre-processing\original-image.png) | ![riginal-image-histogra](.\pre-processing\original-image-histogram.png) |
| **Image - Histogram Equalized**![mage-histogram-equalize](.\pre-processing\image-histogram-equalized.png) | ![uqalized-histogram-cd](.\pre-processing\euqalized-histogram-cdf.png) |
| **Original Image       **![riginal-image-morp](.\pre-processing\original-image-morph.png) | **Morphological Dilation**![mage-morph-dilatio](.\pre-processing\image-morph-dilation.png) |

The original premise is to prevent any bias towards a specific image region with lower or higher intensities, so one method used was to grayscale all images and equalize their histograms.   

Additionally, OpenCV's morphological operations were used to determine whether there were additional features that could be detected.  The above table, in the 3rd row, shows a grayscaled image on the left and a morphological dilated image on the right.  This particular dilation is able to show object edges with significant contrast, such as in the person's forearm, but it becomes difficult to detect edges in uniform colored areas such as the person's upper arm with a black shirt.  So this dilation technique to detect edge features did not seem to be an effective technique and was not used.

### Algorithms and Techniques

The solution will consist of a machine learning pipeline with pre-processing, training, testing, and accuracy measurement stages.  The solution will use Convolutional Neural Networks (CNNs) since the input data is a set of images, i.e. 2-D tensors, and  CNNs have been proven to be very effective for image classification, and in particular for posture classification [5,9].  In the pre-processing stage, the images will be pre-processed to 224x224x3 , with the same aspect ratio, in order to reduce processing time.  The images will be rescaled, and both raw color images and gray scaled images will be used for model training and prediction.  Additionally, the CNN may use pooling to allow for position invariance, *softmax* activation function for classifying based on likelihood since the output will  based on a set of mutually exclusive classes, *ReLU* activation function for increasing the non-linear capacity of the network and possibly use regularization methods such as *Dropout*  or *L1 Regularization* to gain processing efficiency and reduce overfitting.   The output will be chosen based on maximum likelihood of a class and compared with target label, and the classification accuracy percentage will be calculated.  

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


There are a couple of benchmarks that can be used to evaluate the performance of learning model.  The first benchmark that can be used is a basic CNN with a single layer without any additional components such as pooling, dropouts or softmax activation functions.   This will set a baseline for how a simple model will perform.  A secondary benchmark model that can be used are the results obtained in the whitepaper, [5], on the same Statefarm dataset.

The whitepaper entitled, "Realtime Distracted Driver Posture Classification", uses the same Statefarm dataset trained with genetically weighted ensemble of CNNs to obtain a classification accuracy of 95.98%.

CNNs will also be used here with a different design to obtain a classification accuracy and compared to the one in the whitepaper to determine whether the CNN design is good or needs to be improved.  Further research will be done to determine whether a more complex CNN is desirable or an ensemble is more appropriate.

Classification accuracy will be used as a primary metric to evaluate the performance of the trained model.   The accuracy will be simply based on the ratio of the number of images classified accurately to the total number of images.  Each image will be classified accurately if the class identified by the model is the same as the label for the image.  This percentage will be used to compare to the benchmark described above.  Since the training dataset will be defined to be balanced, there is no expected skewness, hence no additional adjustments necessary for evaluating performance.

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_



![aseline-model-accuracie](C:\Users\pushkar\ML\machine-learning\projects\capstone\results\baseline-model-accuracies.PNG)





![odel_evaluation_results_](C:\Users\pushkar\ML\machine-learning\projects\capstone\results\model_evaluation_results_1.PNG)

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?



### Appendix

The following table shows the results of several models trained over 100 epochs with 15,200 training samples, 3800 validation samples and 3800 testing samples.   The results are shown based on the maximum testing accuracy for a particular set of hyperparameter values for a specific model design.

| Model | Model Description                                            | Regularization                  | Training Accuracy | Validation Accuracy | Testing Accuracy | Training vs. Testing accuracy |
| ----- | ------------------------------------------------------------ | ------------------------------- | ----------------- | ------------------- | ---------------- | ----------------------------- |
| 1     | {Conv2D->MaxPooling2D}(3) -> Flatten -> Dense(**relu**) -> **Dropout** -> Dense (**softmax**) | Dropout = .05                   | 94.79%            | 91.84%              | 92.08%           |                               |
| 2     | {Conv2D->MaxPooling2D}(3) -> Flatten -> Dense(relu, **L1**) -> **Dropout** -> Dense (**softmax**) | Dropout=.20, L1 Regularizer=.05 | 10.28%            | 9.39%               | 9.76%            |                               |
| 3     | {Conv2D->MaxPooling2D}(3) -> Flatten -> Dense(**relu**, **L1**) -> Dense (**softmax**) | L1 Regularizer = .05            | 97%               | 91.05%              | 90.37%           |                               |
| 4     | {Conv2D->MaxPooling2D}(3) -> Flatten -> Dense(**softmax**) -> Dropout -> Dense (**softmax**) | Dropout = .30                   | 77.45%            | 92.32%              | 91.95%           |                               |

### References

**Reference**

[1] https://www.kaggle.com/c/state-farm-distracted-driver-detection

[2] https://www.cdc.gov/motorvehiclesafety/distracted_driving/index.html

[3] https://www.nhtsa.gov/risky-driving/distracted-driving

[4] https://www.nhtsa.gov/sites/nhtsa.dot.gov/files/documents/driver_electronic_device_use_in_2015_0.pdf

[5] *Realtime Distracted Driver Posture Classification*, https://arxiv.org/pdf/1706.09498.pdf

[6] https://www.kaggle.com/c/state-farm-distracted-driver-detection/download/imgs.zip

[7] *Benchmark of Deep Learning Models on Large Healthcare MIMIC Datasets*, https://arxiv.org/pdf/1710.08531.pdf

[8] *Metrics To Evaluate Machine Learning Algorithms in Python*, https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/

[9] *Application of Convolutional Neural Network to Classify Sitting and Standing Postures*, http://www.iaeng.org/publication/WCECS2017/WCECS2017_pp140-144.pdf