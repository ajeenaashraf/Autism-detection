import numpy as np 
import pandas as pd
from glob import glob 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

train_path = 'C:/Users/madha/OneDrive/Desktop/autismfinal/Datasets/train'
test_path = 'C:/Users/madha/OneDrive/Desktop/autismfinal/Datasets/test'
IMAGE_SIZE = [224, 224] #Default image size for VGG16
folders = glob('C:/Users/madha/OneDrive/Desktop/autismfinal/Datasets/train/*') #Get number of classes
# ImageDataGenerator can help perform augumentation on existing images. This way, we get more diverse train set.
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
#Through flow_from_directory - we create an array of images that can be used for training. 
training_set = train_datagen.flow_from_directory(
'C:/Users/madha/OneDrive/Desktop/autismfinal/Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')
validation_set = validation_datagen.flow_from_directory('C:/Users/madha/OneDrive/Desktop/autismfinal/Datasets/test',
                                                 target_size = (224, 224),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('C:/Users/madha/OneDrive/Desktop/autismfinal/Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')



# Create a VGG16 model, and removing the last layer that is classifying 1000 + images. This will be replaced with images classes we have. 
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False) #Training with Imagenet weights
# Use this line for VGG19 network. Create a VGG19 model, and removing the last layer that is classifying 1000 images. This will be replaced with images classes we have. 
#vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# This sets the base that the layers are not trainable. If we'd want to train the layers with custom data, these two lines can be ommitted. 
for layer in vgg.layers:
  layer.trainable = False
x = Flatten()(vgg.output) #Output obtained on vgg16 is now flattened. 
prediction = Dense(len(folders), activation='softmax')(x) # We have 2 classes, and so, the prediction is being done on len(folders) - 2 classes
#Creating model object 
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

#Compile the model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
history = model.fit(training_set, validation_data=validation_set, epochs=20, batch_size=32)


print("[INFO] saving  model...")
model.save("Classification.model", save_format="h5")


# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig("plot.png")
 
 # accuracies
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig("plot1.png")


#print("[INFO] saving training model...")
#model.save("Classification.model", save_format="h5")



