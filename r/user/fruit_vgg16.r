# Source: https://shirinsplayground.netlify.com/2018/06/keras_fruits_lime/
# Author: Dr. Shirin Elsinghorst - https://github.com/ShirinG
# Data: https://www.kaggle.com/moltean/fruits/data

library(keras)   # for working with neural nets
library(lime)    # for explaining models
library(magick)  # for preprocessing images
library(ggplot2) # for additional plotting

# Loading model
model <- application_vgg16(weights = "imagenet", include_top = TRUE)
model

# Include our own model
path_prefix <- paste(getwd(), "/data/fruit/", sep="") 
model2 <- load_model_hdf5(paste(path_prefix, "fruit_checkpoints.h5", sep=""))
model2

##################################################################################

# Load images
test_image_files_path <- paste(path_prefix, "validation/", sep="") 

img <- image_read('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/272px-Banana-Single.jpg')
img_path <- file.path(test_image_files_path, "Banana", 'banana.jpg')
image_write(img, img_path)
#plot(as.raster(img))

img2 <- image_read('https://cdn.pixabay.com/photo/2010/12/13/09/51/clementine-1792_1280.jpg')
img_path2 <- file.path(test_image_files_path, "Clementine", 'clementine.jpg')
image_write(img2, img_path2)
#plot(as.raster(img2))

plot_superpixels(img_path, n_superpixels = 35, weight = 10)
plot_superpixels(img_path2, n_superpixels = 50, weight = 20)

##################################################################################

# Prepare images for image net
image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(224,224))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

# Test predictions
res <- predict(model, image_prep(c(img_path, img_path2)))
imagenet_decode_predictions(res)

# Load labels and train explainer

model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
explainer <- lime(c(img_path, img_path2), as_classifier(model, model_labels), image_prep)

# Train model
explanation <- explain(c(img_path, img_path2), explainer, 
                       n_labels = 2, n_features = 35,
                       n_superpixels = 35, weight = 10,
                       background = "white")

plot_image_explanation(explanation)

clementine <- explanation[explanation$case == "clementine.jpg",]
plot_image_explanation(clementine)

##################################################################################

# Make predictions

test_datagen <- image_data_generator(rescale = 1/255)

test_generator <- flow_images_from_directory(
  test_image_files_path,
  test_datagen,
  target_size = c(20, 20),
  class_mode = 'categorical')

predictions <- as.data.frame(predict_generator(model2, test_generator, steps = 1))

load(paste(path_prefix, "fruits_classes_indices.RData", sep=""))
fruits_classes_indices_df <- data.frame(indices = unlist(fruits_classes_indices))
fruits_classes_indices_df <- fruits_classes_indices_df[order(fruits_classes_indices_df$indices), , drop = FALSE]
colnames(predictions) <- rownames(fruits_classes_indices_df)

t(round(predictions, digits = 2))

for (i in 1:nrow(predictions)) {
  cat(i, ":")
  print(unlist(which.max(predictions[i, ])))
}

image_prep2 <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(20, 20))
    x <- image_to_array(img)
    x <- reticulate::array_reshape(x, c(1, dim(x)))
    x <- x / 255
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

fruits_classes_indices_l <- rownames(fruits_classes_indices_df)
names(fruits_classes_indices_l) <- unlist(fruits_classes_indices)
fruits_classes_indices_l

explainer2 <- lime(c(img_path, img_path2), as_classifier(model2, fruits_classes_indices_l), image_prep2)
explanation2 <- explain(c(img_path, img_path2), explainer2, 
                        n_labels = 1, n_features = 20,
                        n_superpixels = 35, weight = 10,
                        background = "white")

# plot results
explanation2 %>%
  ggplot(aes(x = feature_weight)) +
  facet_wrap(~ case, scales = "free") +
  geom_density()
