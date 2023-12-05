import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder()  # Encoder used to translate an image into an array.
encoder.fit([[0], [1], [2], [3], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [18], [19], [20], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32]])
data = []
paths1 = []
paths2 = []
paths3 = []
paths4 = []
paths5 = []
paths6 = []
paths7 = []
paths8 = []
paths9 = []
paths10 = []
paths11 = []
paths12 = []
paths13 = []
paths14 = []
paths15 = []
paths16 = []
paths17 = []
paths18 = []
paths19 = []
paths20 = []
paths21 = []
paths22 = []
paths23 = []
paths24 = []
paths25 = []
paths26 = []
paths27 = []
paths28 = []
paths29 = []
paths30 = []
paths31 = []
paths32 = []
paths33 = []




label = []  # Label array.

# ------------------------------------------------------------------------------
for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Apple_scab'):
    for file in f:
        paths1.append(os.path.join(r, file))  # Full path.

for path in paths1:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[0]]).toarray())
print(len(label))
paths1 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Black_rot'):
    for file in f:
        paths2.append(os.path.join(r, file))  # Full path.

for path in paths2:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[1]]).toarray())
print(len(label))
paths2 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Cedar_apple_rust'):
    for file in f:
        paths3.append(os.path.join(r, file))  # Full path.

for path in paths3:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[2]]).toarray())
print(len(label))
paths3 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___healthy'):
    for file in f:
        paths4.append(os.path.join(r, file))  # Full path.

for path in paths4:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[3]]).toarray())
print(len(label))
paths4 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'):
    for file in f:
        paths7.append(os.path.join(r, file))  # Full path.

for path in paths7:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[6]]).toarray())
print(len(label))
paths7 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Corn_(maize)___Common_rust_'):
    for file in f:
        paths8.append(os.path.join(r, file))  # Full path.

for path in paths8:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[7]]).toarray())
print(len(label))
paths8 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Corn_(maize)___healthy'):
    for file in f:
        paths9.append(os.path.join(r, file))  # Full path.

for path in paths9:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[8]]).toarray())
print(len(label))
paths9 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Corn_(maize)___Northern_Leaf_Blight'):
    for file in f:
        paths10.append(os.path.join(r, file))  # Full path.

for path in paths10:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[9]]).toarray())
print(len(label))
paths10 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Grape___Black_rot'):
    for file in f:
        paths11.append(os.path.join(r, file))  # Full path.

for path in paths11:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[10]]).toarray())
print(len(label))
paths11 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Grape___Esca_(Black_Measles)'):
    for file in f:
        paths12.append(os.path.join(r, file))  # Full path.

for path in paths12:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[11]]).toarray())
print(len(label))
paths12 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Grape___healthy'):
    for file in f:
        paths13.append(os.path.join(r, file))  # Full path.

for path in paths13:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[12]]).toarray())
print(len(label))
paths13 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'):
    for file in f:
        paths14.append(os.path.join(r, file))  # Full path.

for path in paths14:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[13]]).toarray())
print(len(label))
paths14 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Peach___Bacterial_spot'):
    for file in f:
        paths15.append(os.path.join(r, file))  # Full path.

for path in paths15:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[14]]).toarray())
print(len(label))
paths15 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Peach___healthy'):
    for file in f:
        paths16.append(os.path.join(r, file))  # Full path.

for path in paths16:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[15]]).toarray())
print(len(label))
paths16 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Potato___Early_blight'):
    for file in f:
        paths19.append(os.path.join(r, file))  # Full path.

for path in paths19:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[18]]).toarray())
print(len(label))
paths19 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Potato___healthy'):
    for file in f:
        paths20.append(os.path.join(r, file))  # Full path.

for path in paths20:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[19]]).toarray())
print(len(label))
paths20 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Potato___Late_blight'):
    for file in f:
        paths21.append(os.path.join(r, file))  # Full path.

for path in paths21:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[20]]).toarray())
print(len(label))
paths21 = []  # Path array


# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Bacterial_spot'):
    for file in f:
        paths24.append(os.path.join(r, file))  # Full path.

for path in paths24:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[23]]).toarray())
print(len(label))
paths24 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Early_blight'):
    for file in f:
        paths25.append(os.path.join(r, file))  # Full path.

for path in paths25:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[24]]).toarray())
print(len(label))
paths25 = []  # Path array


# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___healthy'):
    for file in f:
        paths26.append(os.path.join(r, file))  # Full path.

for path in paths26:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[25]]).toarray())
print(len(label))
paths26 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Late_blight'):
    for file in f:
        paths27.append(os.path.join(r, file))  # Full path.

for path in paths27:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[26]]).toarray())
print(len(label))
paths27 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Leaf_Mold'):
    for file in f:
        paths28.append(os.path.join(r, file))  # Full path.

for path in paths28:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[27]]).toarray())
print(len(label))
paths28 = []  # Path array


# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Septoria_leaf_spot'):
    for file in f:
        paths29.append(os.path.join(r, file))  # Full path.

for path in paths29:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[28]]).toarray())
print(len(label))
paths29 = []  # Path array


# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Spider_mites Two-spotted_spider_mite'):
    for file in f:
        paths30.append(os.path.join(r, file))  # Full path.

for path in paths30:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[29]]).toarray())
print(len(label))
paths30 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Target_Spot'):
    for file in f:
        paths31.append(os.path.join(r, file))  # Full path.

for path in paths31:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[30]]).toarray())
print(len(label))
paths31 = []  # Path array


# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Tomato_mosaic_virus'):
    for file in f:
        paths32.append(os.path.join(r, file))  # Full path.

for path in paths32:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[31]]).toarray())
print(len(label))
paths32 = []  # Path array

# ------------------------------------------------------------------------------

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Plant_Disease_Detection/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus'):
    for file in f:
        paths33.append(os.path.join(r, file))  # Full path.

for path in paths33:
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape == (224, 224, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[32]]).toarray())
print(len(label))
paths33 = []  # Path array



# Filling data array.
data = np.array(data)
var = data.shape

# Filling label array.
label = np.array(label)
label = label.reshape(49706, 27)  # Returns an array with the same data, to the new shape (Classification).


# Separates the Test from Train data.
x_Train, x_Test, y_Train, y_Test = train_test_split(data, label, test_size=0.1, shuffle=True, random_state=0)


# Separates the Validation from Train data.
x_Train, x_Val, y_Train, y_Val = train_test_split(x_Train, y_Train, test_size=0.25, shuffle=True, random_state=0)

# Saving Test data
np.save('TestData', x_Test)
np.save('TestData2', y_Test)



# Creating a CNN model
model = Sequential()

# INPUT
model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(224, 224, 3), padding='Same'))

# 1st CONV
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='Same'))
model.add(BatchNormalization())  # Normalizes data in batches
model.add(MaxPooling2D(pool_size=(2, 2)))  # Down-samples the data
model.add(Dropout(0.5))  # Randomly sets input units to 0 to help over-fitting.

# 2nd CONV
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))


model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(27, activation='softmax'))

# Model settings.
model.compile(loss="categorical_crossentropy", optimizer='Adamax', metrics=['acc'])
history = model.fit(x_Train, y_Train, epochs=32, batch_size=32, verbose=1, validation_data=(x_Val, y_Val))



# Figure1
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Figure 1')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# Figure2
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.title('Figure 2')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(range(32), acc, label='train')
plt.plot(range(32), val_acc, label='test')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# Saves the model results.
model.save('CNN_data/Plant_disease_Detect.h5')

# Labels
labels = ["Apple_Scab", "Apple_BR", "Apple_rust", "Apple_healthy", "Corn_Cercospora", "Corn_Common_rust", "Corn_healthy", "Corn_NLeafB",
          "Grape_BR", "Grape_Esca", "Grape_healthy", "Grape_leafB", "Peach_BacterialS", "Peach_healthy",
          "Potato_EB", "Potato_healthy", "Potato_LB",
          "Tomato_BS", "Tomato_EB", "Tomato_healthy", "Tomato_LB", "Tomato_leafMold",
          "Tomato_Septoria", "Tomato_SpiderM", "Tomato_TargetS", "Tomato_MosaicV", "TomatoYellow"]

prediction = model.predict(x_Test)


# Figure3- Testing
figure = plt.figure(figsize=(18, 10))

for i, index in enumerate(np.random.choice(x_Test.shape[0], size=18, replace=False)):
    ax = figure.add_subplot(3, 6, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_Test[index]))
    predict_index = np.argmax(prediction[index])
    true_index = np.argmax(y_Test[index])
    ax.set_title("{}".format(labels[predict_index]), color=("green" if predict_index == true_index else "red"))

plt.show()