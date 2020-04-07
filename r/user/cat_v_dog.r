# Source: https://blogs.rstudio.com/tensorflow/posts/2017-12-14-image-classification-on-small-datasets/
# Source: Chollet & Allaire, 2017 - https://github.com/fchollet & https://github.com/jjallaire
# Data: https://www.kaggle.com/c/dogs-vs-cats/data

library(keras)

# Set dirs
base_dir <- "~/data/cats_and_dogs_small"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")
train_cats_dir <- file.path(train_dir, "cats")
train_dogs_dir <- file.path(train_dir, "dogs")
validation_cats_dir <- file.path(validation_dir, "cats")
validation_dogs_dir <- file.path(validation_dir, "dogs")
test_cats_dir <- file.path(test_dir, "cats")
test_dogs_dir <- file.path(test_dir, "dogs")


# Load VGG16 model

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)
summary(conv_base)

model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model)
length(model$trainable_weights)

freeze_weights(conv_base)
length(model$trainable_weights)

####################################################################
# Train model

train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

# Note that the validation data shouldn't be augmented!
test_datagen <- image_data_generator(rescale = 1/255)  

train_generator <- flow_images_from_directory(
  train_dir,                  # Target directory  
  train_datagen,              # Data generator
  target_size = c(150, 150),  # Resizes all images to 150 × 150
  batch_size = 20,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

summary(conv_base)

####################################################################

# Fine tune model
unfreeze_weights(conv_base, from = "block3_conv1")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

####################################################################

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

model %>% evaluate_generator(test_generator, steps = 50)