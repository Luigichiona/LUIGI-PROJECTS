#!/usr/bin/env python
# coding: utf-8

# # Loading Dataset
# 
# You can download the dataset from {https://challenge.isic-archive.com/data/#2018}.
# The data from **Task 3** will be used in this assignment. You should download the Training Data and its corresponding ground truth labels. The dataset consists of 10015 skin images from 7 classes. We will extract the images of 7 classes (Melanoma, Melanocytic nevi, Basal cell carcinoma, Actinic keratoses and intraepithelial carcinoma (akaic), Benign keratosis-like lesions, Dermatofibroma\ and Vascular lesions) and save them as .npy file with the following code:
# 

# In[2]:


from google.colab import drive
drive.mount('/content/drive/',force_remount = True)


# The code below runs really slow in Colab. I have uploaded the resulting data files from the code below for direct access. No need to run this section.

# In[ ]:


import os #  - loaded in section below
import pandas as pd #  - loaded in section below
import numpy as np #  - loaded in section below
from PIL import Image # - loaded in section below
import matplotlib.pyplot as plt #- loaded in section below


# No need to run this section, data files already present in the drive. Takes a long time to complete
# Added a counter so that we won't have to wait approximately an hour or for this code run
# The complete files are present in the drive, but a subset can be used for the sample images.


# Replace these paths with the actual paths to your dataset folders
data_folder = "/content/drive/MyDrive/Deep Learning Assignment/ISIC2018_Task3_Training_Input/"
ground_truth_folder = "/content/drive/MyDrive/Deep Learning Assignment/ISIC2018_Task3_Training_GroundTruth/"

csv_file_path = os.path.join(ground_truth_folder, "ISIC2018_Task3_Training_GroundTruth.csv")
df = pd.read_csv(csv_file_path)

image_data = []
labels = []
counter = 0

for index, row in df.iterrows():
    if counter >= 1000:
      break
    image_title = row['image']
    label = row.drop('image', axis=0)  # Drop the 'image' column to keep only labels

    image_path = os.path.join(data_folder, image_title + ".jpg")

    try:
        # Open the image using PIL (or you can use OpenCV) within a 'with' statement
        with Image.open(image_path) as image:
            if image is not None:
                # Resize images
                im = image.resize((120,150), Image.LANCZOS)
                # Append image and label to respective lists
                image_data.append(np.array(im))
                labels.append(label)
                counter += 1
                if counter  % 100 == 0:
                  print('processed images:',counter)
            else:
                print(f"Error opening image '{image_path}': NoneType object returned")
    except Exception as e:
        print(f"Error opening image '{image_path}': {e}")

tr_labels = np.array(labels)
image_matrix = np.array([np.array(img) for img in image_data])

np.save("/content/drive/MyDrive/Deep Learning Assignment/Assignment/data_subset.npy",image_matrix)
np.save("/content/drive/MyDrive/Deep Learning Assignment/Assignment/labels_subset.npy",tr_labels)

# Class mapping
class_mapping = {
    tuple([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]): "Melanoma",
    tuple([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]): "Melanocytic nevi",
    tuple([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]): "Basal cell carcinoma",
    tuple([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]): "Acaic",
    tuple([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]): "Benign keratosis-like lesions",
    tuple([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]): "Dermatofibroma",
    tuple([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]): "Vascular lesions"
}

# Convert float labels to class names
class_labels = [class_mapping[tuple(label)] for label in tr_labels]
np.save("/content/drive/MyDrive/Deep Learning Assignment/Assignment/labels_name_subset.npy",class_labels)


# Once you save your data, you can load it from your directory.

# In[3]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers

file_dir = "/content/drive/MyDrive/Deep Learning Assignment/Assignment/"
data = np.load(file_dir + "data.npy")
labels = np.load(file_dir + "labels.npy", allow_pickle=True)
class_labels = np.load(file_dir + "labels_name.npy")
lesions = ('Melanoma','Melanocytic nevi','Basal cell carcinoma','Acaic','Benign keratosis-like lesions','Dermatofibroma','Vascular lesions')


# ## Preprocessing

# In[4]:


import tensorflow as tf
import keras
import seaborn as sns
import logging
logging.getLogger('tensorflow').disabled = True


# Checking the shapes of the imported data

# In[ ]:


# Data shapes --------------------------------
print(data.shape[0],'images.','\nResolution:',data.shape[1],'x',data.shape[2],'\nColor channels:',data.shape[3],'\n')
print(labels.shape[0],'classifications','\nUnique ground truths:',labels.shape[1])


# In[ ]:


def plot_history(data_list, label_list, title, ylabel):
    ''' Plots a list of vectors.

    Parameters:
        data_list  : list of vectors containing the values to plot
        label_list : list of labels describing the data, one per vector
        title      : title of the plot
        ylabel     : label for the y axis
    '''
    epochs = range(1, len(data_list[0]) + 1)

    for data, label in zip(data_list, label_list):
        plt.plot(epochs, data, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend(fontsize = 7)

    plt.show()


# We need to split the data into a training, validation and testing set. We will be using 20% of the data as test data, and 20% of the data as validation data. The remaining 60% will be used as training data.

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test_val, y_train, y_test_val = train_test_split(data,labels,test_size= 0.4, random_state= 42, stratify= labels)
X_test, X_val, y_test, y_val = train_test_split(X_test_val,y_test_val,test_size= 0.5, random_state= 42, stratify= y_test_val)

print('Features training data\n',X_train.shape[0],'images','\nResolution:',X_train.shape[1],'x',X_train.shape[2],'\nColor channels:',X_train.shape[3],'\n')
print('Ground truth training data\n',y_train.shape[0],'classifications','\nGround truths:',y_train.shape[1],'\n\n')
print('Features validation data\n',X_val.shape[0],'images','\nResolution:',X_val.shape[1],'x',X_val.shape[2],'\nColor channels:',X_val.shape[3],'\n')
print('Ground truth validation data\n',y_val.shape[0],'classifications','\nGround truths:',y_val.shape[1],'\n\n')
print('Features test data\n',X_test.shape[0],'images','\nResolution:',X_test.shape[1],'x',X_test.shape[2],'\nColor channels:',X_test.shape[3],'\n')
print('Ground truth test data\n',y_test.shape[0],'classifications','\nGround truths:',y_test.shape[1],'\n')


# The images are represented by values ranging from 0 to 255. Each of the values within a channel corresponds to the intensity of the corresponding color that channel represents. Every pixel's color at a specific location within an image is represented by a combination of the intensities of the three channels present.
# 
# Furthermore, these values need to be normalized before we can proceed with training the CNN model.

# In[16]:


X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('float32')
y_val = y_val.astype('float32')
y_test = y_test.astype('float32')

# Also reshape the data so that it can be fed into the CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 3))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2], 3))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 3))

# NOrmalize the image data
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0


# Fixing class imbalance, didn't run it yet so we can change it

# In[ ]:





# # Visualizing Sample images

# In[ ]:


# Plot the images with labels
np.random.seed(95)
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i, (index, row) in enumerate(df.sample(15).iterrows()):
    fname = data_folder + row['image'] + '.jpg'
    image = Image.open(fname)
    tmp = row.drop(['image'])
    label = tmp.index[tmp.values.argmax()]
    ax = axes[i // 5, i % 5]
    ax.imshow(image)
    ax.set_title(label)
    ax.axis('off')
plt.tight_layout()
plt.show()


# # Visualize Class Label Distribution

# In[ ]:


distribution_data = df.drop(columns=['image']).sum(axis=0).to_dict()

labels = list(distribution_data.keys())
heights = list(distribution_data.values())

plt.figure(figsize=(8, 6))
plt.bar(labels, heights, color='lightblue') # Create a bar plot
plt.xlabel('Labels: Skin Conditions')
plt.ylabel('Count')
plt.title('Skin Condition Counts')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Baseline Model

# In[ ]:


get_ipython().system('pip install imbalanced-learn')


# In[7]:


from imblearn.under_sampling import RandomUnderSampler, TomekLinks

#two undersampling techniques which can have different results
# Flatten the data because
num_samples, height, width, channels = X_train.shape
X_train_reshaped = X_train.reshape(num_samples, height * width * channels)

# Apply undersampling
rus = RandomUnderSampler(random_state=42)
# Create a Tomek Links undersampler
tl = TomekLinks(sampling_strategy='auto', n_jobs=-1)

# Fit and transform the data
X_resampled, y_resampled = tl.fit_resample(X_train_reshaped, y_train)
X_res_reshaped, y_res = rus.fit_resample(X_train_reshaped, y_train)

# Reshape the data back to its original shape
X_res = X_res_reshaped.reshape(-1, height, width, channels)
X_res_2 = X_resampled.reshape(-1, height, width, channels)



# In[14]:


#bar graph, does not work as expected, it creates 2 bars instead of 7
#need bar graph after undersampling that shows equally distributed classes

# 1. Count the occurrences of each class
unique_classes_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
unique_classes_res, counts_res = np.unique(y_res, return_counts=True)

# 2. Plot the counts for TomekLinks
plt.figure(figsize=(12, 6))
plt.bar(unique_classes_resampled, counts_resampled, color='b', align='center')
plt.xlabel('Class')
plt.ylabel('Counts')
plt.title('Counts of each class after undersampling with TomekLinks')
plt.xticks(unique_classes_resampled)
plt.tight_layout()
plt.show()

# 3. Plot the counts for RandomUnderSampler
plt.figure(figsize=(12, 6))
plt.bar(unique_classes_res, counts_res, color='r', align='center')
plt.xlabel('Class')
plt.ylabel('Counts')
plt.title('Counts of each class after undersampling with RandomUnderSampler')
plt.xticks(unique_classes_res)
plt.tight_layout()
plt.show()





# In[ ]:


baselineModel= models.Sequential()
#first convolutional layer
baselineModel.add(layers.Conv2D(64, (3,3),activation='relu', input_shape=(150, 120, 3), padding='same'))

baselineModel.add(layers.Conv2D(32, (3,3), activation='relu',padding='same'))

baselineModel.add(layers.MaxPooling2D((2,2)))

#2nd convolutional layer
baselineModel.add(layers.Conv2D(64, (3,3), activation='relu',padding='same'))

baselineModel.add(layers.Conv2D(32, (3,3), activation='relu',padding='same'))

baselineModel.add(layers.MaxPooling2D(pool_size=(2,2)))

#flatten layer w/ (none, 35520)
baselineModel.add(layers.Flatten())

#dense layer w/ (none, 32)
baselineModel.add(layers.Dense(32, activation='relu'))

#dense layer w/ (none, 32)
baselineModel.add(layers.Dense(32, activation='relu'))

#dense layer w/ (none, 7)
baselineModel.add(layers.Dense(7, activation='softmax'))

#compile model
baselineModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(baselineModel.summary())


# # Baseline evaluation

# epoch 10

# In[ ]:


baselineHistory = baselineModel.fit(
    X_res, y_res,
    epochs=10,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose = 1
)

baseline_loss, baseline_acc = baselineModel.evaluate(X_val,y_val)

acc_baseline = [baselineHistory.history['accuracy'],baselineHistory.history['val_accuracy']]

acc_labels_baseline = ['Baseline Training Accuracy','Baseline Validation Accuracy ']

loss_baseline = [baselineHistory.history['loss'],baselineHistory.history['val_loss']]

loss_labels_baseline = ['Baseline Training Loss','Baseline Validation Loss']

plot_history(acc_baseline,acc_labels_baseline,'Model Performance','Accuracy')
plot_history(loss_baseline,loss_labels_baseline,'Model Performance','Loss')


# epoch 20

# In[ ]:


baselineHistory = baselineModel.fit(
    X_res, y_res,
    epochs=20, #This is a hyperparameter
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose = 0
)
baseline_loss, baseline_acc = baselineModel.evaluate(X_val,y_val)

acc_baseline = [baselineHistory.history['accuracy'],baselineHistory.history['val_accuracy']]

acc_labels_baseline = ['Baseline Training Accuracy','Baseline Validation Accuracy ']

loss_baseline = [baselineHistory.history['loss'],baselineHistory.history['val_loss']]

loss_labels_baseline = ['Baseline Training Loss','Baseline Validation Loss']

plot_history(acc_baseline,acc_labels_baseline,'Model Performance','Accuracy')
plot_history(loss_baseline,loss_labels_baseline,'Model Performance','Loss')


# In[ ]:


# Python's garbage collector to free up RAM in colab
import gc
gc.collect()


# Confusion matrix of the baseline model predictions against the test dataset

# In[ ]:


from sklearn.metrics import confusion_matrix
# BASELINE CONFUSION MATRIX
pred_baseline_prob = baselineModel.predict(X_test)
pred_baseline = np.argmax(pred_baseline_prob,axis = 1)
pred_baseline = pred_baseline.reshape(-1,1)
yt_baseline = np.argmax(y_test,axis = 1)
yt_baseline = yt_baseline.reshape(-1,1)

confdata_baseline = confusion_matrix(yt_baseline,pred_baseline)
confusionmat_baseline = pd.DataFrame(confdata_baseline)
plt.figure(figsize = (10,7))
sns.heatmap(confusionmat_baseline, annot= True, fmt = 'g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Baseline model')
plt.show()


# In[ ]:


#bar graph class balance

distribution_data2 = df.drop(columns=['image']).sum(axis=0).to_dict()

labels = list(distribution_data2.keys())
heights = list(distribution_data2.values())

plt.figure(figsize=(8, 6))
plt.bar(labels, heights, color='lightblue') # Create a bar plot
plt.xlabel('Labels: Skin Conditions')
plt.ylabel('Count')
plt.title('Skin Condition Counts after downsampling')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Multiclass ROC Curve with microaveraged AUC

# In[ ]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
y_prob_baseline = baselineModel.predict(X_test)


colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])

# Plot for baseline
plt.figure(figsize=(8, 6))
for i, color in zip(range(y_test_binarized.shape[1]), colors):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_prob_baseline[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of {lesions[i]} (area = {roc_auc:0.2f})')
fpr_micro, tpr_micro, _ = roc_curve(y_test_binarized.ravel(), y_prob_baseline.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='deeppink', lw=2, label=f'Micro-average ROC curve (area = {roc_auc_micro:0.2f})')


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class - baseline Model')
plt.legend(loc="lower right", fontsize = 7)
plt.show()


# Classification Report along with performance metrics

# In[ ]:


from sklearn.metrics import classification_report
y_test_baseline = np.argmax(y_test,axis = 1)
y_pred_baseline = np.argmax(y_prob_baseline, axis = 1)
print(classification_report(y_test_baseline, y_pred_baseline, target_names= lesions))


# ## Enhanced Model

# 

# Finding the right hyperparameters to enhance the baseline model. Therefore, using the RandomSearch to automatically find the best hyperparameters.build_model function is structured based on the baselineModel, but with added hyperparameter tuning capabilities using Keras Tuner's functionalities.

# In[ ]:


import shutil
shutil.rmtree('/content/drive/MyDrive/Deep Learning Assignment/Assignment/Enhanced_model', ignore_errors=True)


# In[ ]:


get_ipython().system('pip install keras-tuner # if not loaded')


# In[ ]:


from keras_tuner import RandomSearch
from tensorflow.keras import layers, models, regularizers


def build_model(hp):
    model = models.Sequential()

    # Optimize number of filters and kernel size for Conv layers
    model.add(layers.Conv2D(filters=hp.Int('conv_filters', min_value = 32, max_value = 128, step = 16),
                            kernel_size=hp.Choice('conv_kernel_size', [3, 5]),
                            activation=hp.Choice('activation_function', ['relu', 'tanh']),
                            padding='same',
                            input_shape=(150, 120, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation=hp.Choice('activation_function', ['relu', 'tanh'])))
    model.add(layers.Dropout(rate=hp.Float('dropout', 0.0, 0.5, 0.1)))

    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

tuner_search = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='/content/drive/MyDrive/Deep Learning Assignment/Assignment/',
    project_name='Enhanced_model')

# For demonstration, I've set epochs to 20; adjust as necessary
tuner.search(X_res, y_res, epochs=20, validation_split=0.2)

enhanced_model = tuner.get_best_models(num_models=1)[0]




# In[ ]:


tuner.search(X_train, y_train, epochs=10, validation_data = (X_val,y_val))


# In[ ]:


modelsearch = tuner.get_best_models(num_models=1)[0]
modelsearch.save('/content/drive/MyDrive/Deep Learning Assignment/Assignment/Enhanced_model/modelsearch.h5')
modelsearch.summary()


# Train model

# In[ ]:


enhancedModel = models.load_model('/content/drive/MyDrive/Deep Learning Assignment/Assignment/Enhanced_model/modelsearch.h5')
enhancedHistory = enhancedModel.fit(X_train,y_train, epochs = 10, validation_data = (X_val,y_val), batch_size = 32)


# In[ ]:


enhancedModel = models.load_model('/content/drive/MyDrive/Deep Learning Assignment/Assignment/Enhanced_model/modelsearch.h5')
enhancedHistory = enhancedModel.fit(X_train,y_train, epochs = 15, validation_data = (X_val,y_val), batch_size = 32)


# Plotting the ROC Curve with AUC Score Baseline VS New model
# 
# 

# In[ ]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize


# Predict probabilities for the baseline and new models
y_prob_baseline = baselineModel.predict(X_val)
y_prob_new = enhancedModel.predict(X_val)

# Binarize the labels
y_bin_val = label_binarize(y_val, classes=[0, 1, 2, 3, 4, 5, 6])
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])

# Compute ROC curve and ROC area for each class for both models
fpr_baseline = dict()
tpr_baseline = dict()
roc_auc_baseline = dict()

fpr_new = dict()
tpr_new = dict()
roc_auc_new = dict()

for i in range(7):
    fpr_baseline[i], tpr_baseline[i], _ = roc_curve(y_bin_val[:, i], y_prob_baseline[:, i])
    roc_auc_baseline[i] = auc(fpr_baseline[i], tpr_baseline[i])

    fpr_new[i], tpr_new[i], _ = roc_curve(y_bin_val[:, i], y_prob_new[:, i])
    roc_auc_new[i] = auc(fpr_new[i], tpr_new[i])

# Plot for Baseline
plt.figure(figsize=(8, 6))
for i, color in zip(range(y_bin_val.shape[1]), colors):
    fpr, tpr, _ = roc_curve(y_bin_val[:, i], y_prob_baseline[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of {lesions[i]} (area = {roc_auc:0.2f})')
fpr_micro, tpr_micro, _ = roc_curve(y_bin_val.ravel(), y_prob_baseline.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='deeppink', lw=2, label=f'Micro-average ROC curve (area = {roc_auc_micro:0.2f})')


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class - baseline Model')
plt.legend(loc="lower right", fontsize = 7)
plt.show()

# Resetting color cycle for new plot
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])

# Plot for Enhanced
plt.figure(figsize=(8, 6))
for i, color in zip(range(y_bin_val.shape[1]), colors):
    fpr, tpr, _ = roc_curve(y_bin_val[:, i], y_prob_new[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of {lesions[i]} (area = {roc_auc:0.2f})')
fpr_micro, tpr_micro, _ = roc_curve(y_bin_val.ravel(), y_prob_new.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='deeppink', lw=2, label=f'Micro-average ROC curve (area = {roc_auc_micro:0.2f})')


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class - Enhanced Model')
plt.legend(loc="lower right", fontsize = 7)
plt.show()


# In[ ]:


# ENHANCED MODEL CONFUSION MATRIX
pred_enhanced_prob = enhancedModel.predict(X_test)
pred_enhanced = np.argmax(pred_enhanced_prob,axis = 1)
pred_enhanced = pred_enhanced.reshape(-1,1)
yt_enhanced = np.argmax(y_test,axis = 1)
yt_enhanced = yt_enhanced.reshape(-1,1)

confdata_enhanced = confusion_matrix(yt_enhanced,pred_enhanced)
confusionmat_enhanced = pd.DataFrame(confdata_enhanced)
plt.figure(figsize = (10,7))
sns.heatmap(confusionmat_enhanced, annot= True, fmt = 'g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Enhanced model')
plt.show()


# In[ ]:


print('Enhanced classification performance metrics')
print(classification_report(yt_enhanced, pred_enhanced, target_names= lesions, zero_division = 0.0))


# ## Transfer Learning Model

# Import all the necessary functions from Keras.

# In[ ]:


from tensorflow.keras import Sequential, layers
from sklearn.preprocessing import label_binarize
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle


# The code below shows a VGG16 based transfer learning model. Retraining its pretrained layers is prevented by freezing them as shows in the for loop, where trainable for each layer is set to false. This allows us to use this architecture as a feature extractor.
# 
# Additional layers are added on top of the VGG16 architecture by adding fully connected layers with a relu activation functiong, while also applying 50% dropout. A softmax is applied for the muilti-class classification task.

# In[ ]:


# Introduce the model
vgg_model = VGG16()

# Exclude the top classificaiton layer and define input
vgg_model = VGG16(include_top= False, input_shape=(150,120,3))
# Freeze the layers
for layer in vgg_model.layers:
    layer.trainable = False

# Flatten the final layer
flat_vg = layers.Flatten()(vgg_model.layers[-1].output)
class_vg_dp = layers.Dropout(0.5)(flat_vg)
class_vg_a = layers.Dense(128, activation = 'relu')(class_vg_dp)
class_vg_b = layers.Dense(64, activation = 'relu')(class_vg_a)
output_vg = layers.Dense(7, activation = 'softmax')(class_vg_b)
vgg_model = Model(inputs=vgg_model.inputs, outputs = output_vg)
vgg_model.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
history_vgg = vgg_model.fit(X_train, y_train, validation_data = (X_val,y_val), epochs = 15, verbose = 1)
vgg_model.save("/content/drive/MyDrive/Deep Learning Assignment/Assignment/VGG16/vgg_model.h5")
history_vgg = pd.DataFrame(history_vgg.history)
history_vgg.to_csv('/content/drive/MyDrive/Deep Learning Assignment/Assignment/VGG16/history_vgg.csv')
val_loss_vgg, val_acc_vgg = vgg_model.evaluate(X_val,y_val)
print(val_acc_vgg)


# In[ ]:


# The VGG-16 model can be loaded from the drive here
vgg_model = models.load_model("/content/drive/MyDrive/Deep Learning Assignment/Assignment/VGG16/vgg_model.h5")
vgg_history = pd.read_csv("/content/drive/MyDrive/Deep Learning Assignment/Assignment/VGG16/history_vgg.csv")
vgg_history = vgg_history.to_dict(orient = 'list')
val_loss_vgg, val_acc_vgg = vgg_model.evaluate(X_val,y_val)
print(val_acc_vgg)


# Using the code for the VGG16 code as a template for the DenseNet-121 architecture, we build an identical top layer for ResNet-50's top layer, making it viable for a classification task.

# In[ ]:


import gc
gc.collect()


# In[ ]:


dn_model = DenseNet121()
dn_model = DenseNet121(include_top= False, input_shape=(150,120,3))
for layer in dn_model.layers:
    layer.trainable = False
flat_dn = layers.Flatten()(dn_model.layers[-1].output)
class_dn = layers.Dense(256, activation = 'relu')(flat_dn)
class_dn_dp = layers.Dropout(0.5)(class_dn)
class_dn_a = layers.Dense(128, activation = 'relu')(class_dn_dp)
class_dn_b = layers.Dense(64, activation = 'relu')(class_dn_a)
output_dn = layers.Dense(7, activation = 'softmax')(class_dn_b)
dn_model = Model(inputs=dn_model.inputs, outputs = output_dn)
dn_model.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
history_dn = dn_model.fit(X_train, y_train, validation_data = (X_val,y_val), epochs = 15, verbose = 1)
dn_model.save("/content/drive/MyDrive/Deep Learning Assignment/Assignment/DenseNet/dn_model.h5")
history_dn = pd.DataFrame(history_dn.history)
history_dn.to_csv('/content/drive/MyDrive/Deep Learning Assignment/Assignment/DenseNet/history_dn.csv')
val_loss_dn, val_acc_dn = dn_model.evaluate(X_val,y_val)


# In[ ]:


# The DENSENET-121 model can be loaded from the drive here
dn_model = models.load_model("/content/drive/MyDrive/Deep Learning Assignment/Assignment/DenseNet/dn_model.h5")
dn_history = pd.read_csv("/content/drive/MyDrive/Deep Learning Assignment/Assignment/DenseNet/history_dn.csv")
dn_history = dn_history.to_dict(orient = 'list')
val_loss_dn, val_acc_dn = dn_model.evaluate(X_val,y_val)
print(val_acc_dn)


# In[ ]:


acc_models = [vgg_history['accuracy'],
              vgg_history['val_accuracy'],
              dn_history['accuracy'],
              dn_history['val_accuracy']]

acc_labels = ['VGG-16 Training Accuracy',
              'VGG-16 Validation Accuracy ',
              'DenseNet-121 Training Accuracy',
              'DenseNet-121 Validation Accuracy']

loss_models = [vgg_history['loss'],
              vgg_history['val_loss'],
              dn_history['loss'],
              dn_history['val_loss']]

loss_labels = ['VGG-16 Training Loss',
              'VGG-16 Validation Loss',
              'DenseNet-121 Training Loss',
              'DenseNet-121 Validation Loss']

plot_history(acc_models,acc_labels,'Model Performance','Accuracy')
plot_history(loss_models,loss_labels,'Model Performance','Loss')


# In[ ]:


# VGG16 CONFUSION MATRIX
pred_vgg_prob = vgg_model.predict(X_test)
pred_vgg = np.argmax(pred_vgg_prob,axis = 1)
pred_vgg = pred_vgg.reshape(-1,1)
yt_vgg = np.argmax(y_test,axis = 1)
yt_vgg = yt_vgg.reshape(-1,1)

confdata_vgg = confusion_matrix(yt_vgg,pred_vgg)
confusionmat_vgg = pd.DataFrame(confdata_vgg)
plt.figure(figsize = (10,7))
sns.heatmap(confusionmat_vgg, annot= True, fmt = 'g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('VGG16 model')
plt.show()


# In[ ]:


# DENSENET121 CONFUSION MATRIX
pred_dn = dn_model.predict(X_test)
pred_dn = np.argmax(pred_dn,axis = 1)
pred_dn = pred_dn.reshape(-1,1)
yt_dn = np.argmax(y_test,axis = 1)
yt_dn = yt_dn.reshape(-1,1)

confdata_dn = confusion_matrix(yt_dn,pred_dn)
confusionmat_dn = pd.DataFrame(confdata_dn)
plt.figure(figsize = (10,7))
sns.heatmap(confusionmat_dn, annot= True, fmt = 'g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('ResNet-50 model')
plt.show()


# In[ ]:


y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
y_prob_vgg = vgg_model.predict(X_test)
y_prob_dn = dn_model.predict(X_test)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])

# Plot for VGG
plt.figure(figsize=(8, 6))
for i, color in zip(range(y_test_bin.shape[1]), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_vgg[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of class {lesions[i]} (area = {roc_auc:0.2f})')
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob_vgg.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='deeppink', lw=2, label=f'Micro-average ROC curve (area = {roc_auc_micro:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class - VGG Model')
plt.legend(loc="lower right", fontsize = 7)
plt.show()

# Plot for DenseNet
plt.figure(figsize=(8, 6))
for i, color in zip(range(y_test_bin.shape[1]), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_dn[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of class {lesions[i]} (area = {roc_auc:0.2f})')
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob_dn.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='deeppink', lw=2, label=f'Micro-average ROC curve (area = {roc_auc_micro:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class - DenseNet Model')
plt.legend(loc="lower right", fontsize = 7)
plt.show()


# In[ ]:


y_pred_vgg = np.argmax(y_prob_vgg, axis = 1)
y_pred_dn = np.argmax(y_prob_dn, axis = 1)
y_test_tl = np.argmax(y_test,axis = 1)
print('VGG-16 classification performance metrics')
print(classification_report(y_test_tl, y_pred_vgg, target_names= lesions, zero_division = 0.0))
print('DenseNet-121 classification performance metrics')
print(classification_report(y_test_tl, y_pred_dn, target_names= lesions,zero_division = 0.0))

