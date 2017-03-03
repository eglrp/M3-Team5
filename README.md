# Machine learning for image classification
Master in Computer Vision (UAB) - M3 Machine Learning

Implementation of machine learning and deep learning techniques for image classification.

The code of the different methods is organised in directories.

- [Session 1: SVM classifier](Code/Session1/): A chosen descriptor (SIFT) computes the features of the images that are classified by a SVM.
- [Session 2: Bag Of Visual Words](Code/Session2/)
- [Session 3: Fisher Vectors](Code/Session3/)
- [Session 4: CNN features + SVM](Code/Session4-noMP/): Two approaches considered:
 - Using the already trained VGG model, the features of the last fully connected layer are provided to the SVM classifier.
 - Use the Bag Of Visual Words approach taking as input the features from an inner layer of the VGG model.
- [Session 5: Fine-tune CNN](Code/Session5/):
 - Change the last fully connected layer of the already trained VGG model to match the number of classes of our dataset.
 - Take the output of the previous convolutional layer of the VGG model and add fully connected layers, getting a more compact network.
- [Session 6: CNN from scratch](Code/Session6/): Train a proposed CNN model from scratch.


## Contributors

 * [Idoia Ruiz](https://github.com/idoiaruiz)
 * [Roque Rodriguez](https://github.com/RoqueRouteiral)
 * [Lidia Talavera](https://github.com/LidiaTalavera)
 * [Onofre Martorell](https://github.com/OnofreMartorell)
