
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense

# 가중치 저장 경로
weight_path = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\models\\Resnet50_weights.h5'
top_model_weights_path = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\models\\Resnet50_fc_model.h5'

# 이미지 사이즈
train_data_dir = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\train'
test_date_dir = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\test'
nb_train_samples = 228
nb_test_samples = 96
epochs = 10
batch_size = 16

# build the Resnet network
model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
print('Model loaded')

# build a classifier model to put on top of convolution model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(1, activation='sigmoid'))


# 사전에 학습된 fully_trained로 시작해야함
# classifier, including the top classifier
# in order to successfully do fine-tuning

top_model.load_weights(top_model_weights_path)

# add the model on top of the convolution base
model = Model(input=model.input, output=top_model(model.output))
# model.add(top_model)

# set the first 25 layers (( 마지막 conv block)
# to - non-trianable ( weight will not be updated)
for layer in model.layers[:-1]:
    layer.trainable = False

# complie the model with a SGD/ momentun optimizer
# and a very slow learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    test_date_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_test_samples)
