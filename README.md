Machine Learning and Computer Vision had quite a bit of breakthroughs in the farming sector in the last couple of years. There is no doubt how essential different crops are to our livelihood and their diseases are a major threat to food security. Rapid identification of crop diseases reamins difficult around the globe with only novel, made-for-smartphone applications utilizing Computer Vision and Deep learning fighting back.



IN THE BOX:

1. CNN trained on 27 different classes and close to 50 000 image samples. The dataset used can be found here: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/ , it is an augmented version of a pre-existing dataset and is categorized into 38 different classes. Due to computing limitations and processing power, this was lowered to 27 classes.
-Training and validation ratio were split 75/35, and 90/10 on training and test ratios.
- Following is a list of classes: Apple Scab, Apple Black Rot, Apple rust, Apple healthy, Corn Cercospora, Corn Common rust, Corn healthy, Corn Corthern Leaf Blight, Grape Black rot, Grape Esca, Grape healthy, Grape Leaf Blight, Peach Bacterial spot, Peach healthy, Potato Early Blight, Potato Late Blight, Potato healthy, Tomato Black Spot, Tomato Early Blight, Tomato healthy, Tomato Late Blight, Tomato Leaf Mold. Tomato Septoria leaf spot, Tomato Yellow Curl Virus, Tomato Spider mites, Tomato Target spot, Tomato Mosaic Virus
- CNN accuracy reached 96% on test data. Tests were conducted on 100 images of each class (resulting in 0-8 mistakes per class, per 100 files).

2. GUI made using Tkinter library with an added splash screen video


*Disclaimer: There is a missing TestData.npy file which was not uploaded due to a large file size and github limitations of the new git LFS (Large File Storage) system.
