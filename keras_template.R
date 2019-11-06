library(tidyverse)
library(keras)
library(fastDummies)
library(tensorflow)


df <- read.csv('~/RStudio/Scripts/TensorFlow/kyphosis.csv')
head(df)
#   Convert First Column to Binary

dummy_data <- fastDummies::dummy_cols(df,remove_first_dummy = TRUE)
head(dummy_data)   
head(df)

#   Remove first Column from Data Set
keep <- c('Age','Number','Start','Kyphosis_present')
final <- dummy_data[keep]
head(final)

#   Split the data into a training and testing set.
library(caret)
index <- createDataPartition(final$Kyphosis_present, p=0.7, list=FALSE)
final.training <- final[index,]
final.test <- final[-index,]

#   Scale the Training Data
x_train <- final.training %>% 
  select(-Kyphosis_present) %>% 
  scale()

y_train <- to_categorical(final.training$Kyphosis_present)

#   Scale the Test Data
x_test <- final.test %>% 
  select(-Kyphosis_present) %>% 
  scale()

y_test <- to_categorical(final.test$Kyphosis_present)

#  We also add drop-out layers to fight overfitting in our model. Similar to Keras in Python, we then add the output layer with the sigmoid activation function. The next step is to compile the model using the binary_crossentropy loss function. This is because we’re solving a binary classification problem. We’ll use the adam optimizer for gradient descent and use accuracy for the metrics. We then fit our model to the training and testing set. Our model will run on 100 epochs using a batch size of 5 and a 30% validation split.
model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'sigmoid')
model

history <- model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


# Fit the Model
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.3,
  callbacks = callback_tensorboard("logs/run_c")
) 


# Lunch TensorBoard
tensorboard("logs/run_c")

# The next thing we can do is evaluate the model. We’ll check the training loss and its accuracy
model %>% evaluate(x_test, y_test)

# Plot the Model 
plot(history)

# Make Predictions
predictions <- model %>% predict_classes(x_test)

# Confusion Matrix
table(factor(predictions, levels=min(final.test$Kyphosis_present):max(final.test$Kyphosis_present)),factor(final.test$Kyphosis_present, levels=min(final.test$Kyphosis_present):max(final.test$Kyphosis_present)))


