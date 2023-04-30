Bias in SVM
This project explores the issue of bias in Support Vector Machines (SVM) and proposes solutions to mitigate it. The notebook Bias_in_SVM.ipynb contains a detailed analysis of the problem and the proposed methods.

Table of contents
Background
Methods
Results
Usage
Contributing
License
Background
SVM is a popular algorithm for classification and regression tasks. It works by finding the hyperplane that maximizes the margin between two classes of data. However, SVM is sensitive to imbalanced datasets and can introduce bias towards the majority class. This is a common problem in real-world applications where one class may be more prevalent than the other.

Methods
The notebook explores two methods to mitigate bias in SVM: class weighting and sampling. Class weighting assigns different weights to each class to balance the contribution of the minority and majority classes. Sampling, on the other hand, involves generating synthetic data points to increase the representation of the minority class.

Results
The notebook shows that both methods are effective in reducing bias in SVM. The performance of each method is evaluated on a synthetic dataset and a real-world dataset. The results demonstrate that class weighting and sampling can improve the accuracy and F1 score of SVM on imbalanced datasets.

Usage
To use the code in this project, simply download or clone the repository and open the Bias_in_SVM.ipynb notebook in Jupyter or Google Colab. The notebook contains detailed instructions on how to run the code and reproduce the results.

Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome any suggestions or improvements to the code or the analysis.

License
This project is licensed under the MIT License - see the LICENSE file for details.
