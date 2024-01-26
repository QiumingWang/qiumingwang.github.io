---
title: 关于missing label中的一些术语
date: 2024-01-14 19:33:46 +8000
categories:
  - det
tags:
  - terms
---
source: [quora](https://www.quora.com/Will-missing-labels-in-training-data-of-object-detection-cause-the-degradation-of-accuracy-Say-object-detection-of-peanuts-I-only-annotate-few-peanuts-in-each-training-image-would-those-unlabeled-peanuts-negatively)

Despite the best efforts to automate annotations, the data annotation process is still partially a manual task, performed by humans with varying experience, ranging from casual people in crowd annotation (Scale) projects to labeling experts in dedicated annotation companies ([Tasq](https://www.tasq.ai/ "www.tasq.ai")). Most mistakes in annotation are therefore caused by humans.

The most common object annotation errors

- Incorrect class: An object is classified incorrectly, e.g. a vehicle is labeled as pedestrian.
- Incorrect attribute: The state of an object is not described correctly, e.g. a car in motion is labeled as parked.
- ==Missing annotation==: An object is not annotated even though it should be.
- Redundant annotation: An object is annotated even though it shouldn’t be
- Incorrect annotation size: An object is not annotated precisely enough, not fitting to its actual dimensions.
- Incorrect annotation position: An object is not annotated precisely enough, not placed at its actual position.

In literature, incorrect classes are generally defined as class noise (Zhu and Wu, 2004), or label noise (Frenay and Verleysen, 2014). For mislabeled classes, the experiment of Fard et al. from 2017 sees a clear dependency on whether the class is mislabeled in an unbiased or biased way.

- Unbiased mislabeling is defined as “random” mislabeling with an equal likelihood that the class is accidentally replaced by any other class.
- Biased mislabeling happens when the annotator confuses the class with always the same class, which induces a constant replacement.

The experiment showed that a) mislabeling in general has a negative impact on performance and b) biased mislabeling has a greater impact on degrading classification performance than unbiased mislabeling. Fard et al. performed the experiment with two models, one convolutional neural network (CNN) and one multi layer perceptron (MLP), whereas the CNN performed better, especially in unbiased mislabeling.

An experiment of Flatow and Penner (2017) examined mislabeling / subjective labeling and its impact on CNN’s accuracy. The results suggest a linear correlation between class noise and test accuracy, where an additional 10% of noise leads to a 4% reduction in accuracy. Further experiments in literature concluded a negative impact of class noise on other machine learning algorithms as well, e.g. impact on decision trees, support vector machines and k nearest neighbors (knn)

