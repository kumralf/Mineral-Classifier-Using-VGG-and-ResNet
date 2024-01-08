import tensorflow as tf
from tensorflow.keras.applications import ResNet50
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
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#for layer in base_model.layers[:51]:
 #   layer.trainable = False
    
#for i, layer in enumerate(base_model.layers):
 #   print (i, layer.name, "-", layer.trainable)
#%%
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
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
#%%
model.save("C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/model_mineral_resnet50_nofreeze.h5")

#%%
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('resnet50_loss_nofreeze.jpg')

#%%
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('resnet50_accuracy_nofreeze.jpg')
#%%

from keras.models import load_model

model_nofreeze = load_model("C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/resnet50/model_mineral_resnet50_nofreeze.h5")
model_first51freeze = load_model("C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/resnet50/model_mineral_resnet50_first51freeze.h5")
model_first100freeze = load_model("C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/resnet50/model_mineral_resnet50_first100freeze.h5")
model_first165freeze = load_model("C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/resnet50/model_mineral_resnet50_first165freeze.h5")

#%%

test_image_path = "C:/Users/kumralf/Desktop/ITU_YL/Computer Vision/Proje/minerals/test/Ruby/ruby_3.jpg"

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
nofreeze = model_nofreeze.evaluate(train_generator)
nofreeze = model_nofreeze.evaluate(validation_generator)
nofreeze = model_nofreeze.evaluate(test_generator)

first51freeze = model_first51freeze.evaluate(train_generator)
first51freeze = model_first51freeze.evaluate(validation_generator)
first51freeze = model_first51freeze.evaluate(test_generator)

first100freeze = model_first100freeze.evaluate(train_generator)
first100freeze = model_first100freeze.evaluate(validation_generator)
first100freeze = model_first100freeze.evaluate(test_generator)

first165freeze = model_first165freeze.evaluate(train_generator)
first165freeze = model_first165freeze.evaluate(validation_generator)
first165freeze = model_first165freeze.evaluate(test_generator)
