import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#%%

train_dir = "C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/train"
validation_dir = "C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/validation"
test_dir = "C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/test"

#%%
batch_size = 16
img_height = 224
img_width = 224
num_classes=87

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

#%%
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


for layer in base_model.layers[:11]:
    layer.trainable = False
    
for i, layer in enumerate(base_model.layers):
    print (i, layer.name, "-", layer.trainable)


#%%

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(87, activation='softmax'))


model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
#%%
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=300,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)

model.save("C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/model_mineral_vgg16_conv4_5_nofreeze.h5")
#%%

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('vgg16_loss_conv4_5_nofreeze.jpg')

plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('vgg16_accuracy_conv4_5_nofreeze.jpg')

#%%

from keras.models import load_model

model = load_model("C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/vgg16/model_mineral_vgg16_conv4_5_nofreeze.h5")

#%%
test_image_path = "C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/test/Bixbite/bixbite_3.jpg"

img = image.load_img(test_image_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale to [0, 1]

predictions = model.predict(img_array)

predicted_class_index = np.argmax(predictions)
predicted_class_label = list(test_generator.class_indices.keys())[predicted_class_index]
confidence = predictions[0][predicted_class_index] * 100

print(f'Tahmin Edilen Sınıf: {predicted_class_label}')
print(f': {confidence:.2f}%')

#%%
xtrain = model.evaluate(train_generator)
xval = model.evaluate(validation_generator)
xtest = model.evaluate(test_generator)