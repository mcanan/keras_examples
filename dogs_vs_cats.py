from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense

# Instancio vgg16 con los pesos de imagenet
vgg16 = VGG16(weights='imagenet')

# Me creo un modelo igual a VGG con la diferencia de la ultima capa.
# En vez de utilizar la ultima capa que clasifica entre 1000 clases de objetos
# utilizo una capa propia que clasifica entre gatos y perros (0=gatos y 1=perros). 
block5_pool = vgg16.get_layer('block5_pool').output
x = Flatten(input_shape=(7,7,512))(block5_pool)
x = Dense(256, activation='relu')(x)
x = Dropout(0.6)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(input=vgg16.input, output=x)

# Generator para entrenamiento
train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True)

# Generator para validacion
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=50,
        class_mode='binary', shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
        'data/validation',
        target_size=(224, 224),
        batch_size=50,
        class_mode='binary', shuffle=False)

for layer in model.layers[:18]:
        layer.trainable = False

model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
model.summary()

model.fit_generator(
        train_generator,
        steps_per_epoch= 4000 // 50,
        epochs=50,
        validation_data=validation_generator,
        validation_steps= 1600 // 50)

model.save_weights('w_dogs_cats.h5')
