##################################################
# ECON 418-518 Homework 3
# Zirui Yan
# The University of Arizona
# ziruiyan@arizona.edu 
# 4 December 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load required libraries
if (!require(pacman)) install.packages("pacman")
pacman::p_load(caret, dplyr,  glmnet, data.table, ISLR2, randomForest)

# Set sead
set.seed(418518)


#####################
# Problem 1
#####################

# Load the dataset
data <- read.csv("~/Econ 418/HW3/ECON_418-518_HW3_Data.csv")

#################
# Question (i)
#################

# Drop columns
data <- data %>%
  select(-fnlwgt, -occupation, -relationship, -capital.gain, -capital.loss, -educational.num)

# Check data
glimpse(data)

#################
# Question (ii)
#################

##############
# Part (a)
##############

# Convert the "income" column to a binary indicator
data$income <- ifelse(data$income == ">50K", 1, 0)


##############
# Part (b)
##############

# Convert the "race" column to a binary indicator
data$race <- ifelse(data$race == "White", 1, 0)

##############
# Part (c)
##############

# Convert the "gender" column to a binary indicator
data$gender <- ifelse(data$gender == "Male", 1, 0)

##############
# Part (d)
##############

# Convert the "workclass" column to a binary indicator
data$workclass <- ifelse(data$workclass == "Private", 1, 0)

##############
# Part (e)
##############

# Convert the "native.country" column to a binary indicator
data$native.country <- ifelse(data$native.country == "United-States", 1, 0)

##############
# Part (f)
##############

# Convert the "marital.status" column to a binary indicator
data$marital.status <- ifelse(data$marital.status == "Married-civ-spouse", 1, 0)

##############
# Part (g)
##############

# Convert the "education" column to a binary indicator
data$education <- ifelse(data$education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)

##############
# Part (h)
##############

# Create the "age_sq" variable as age squared
data$age_sq <- data$age^2

##############
# Part (i)
##############

# Standardize the "age" variable
data$age_standardized <- (data$age - mean(data$age)) / sd(data$age)

# Standardize the "age_sq" variable
data$age_sq_standardized <- (data$age_sq - mean(data$age_sq)) / sd(data$age_sq)

# Standardize the "hours.per.week" variable
data$hours_per_week_standardized <- (data$hours.per.week - mean(data$hours.per.week)) / sd(data$hours.per.week)

#Check the resulting data
glimpse(data)


#################
# Question (iii)
#################

##############
# Part (a)
##############

#  Calculate the proportion of individuals with income greater than $50,000
mean(data$income == 1)

##############
# Part (b)
##############

# Calculate the proportion of individuals in the private sector
mean(data$workclass == 1)

##############
# Part (c)
##############

# Calculate the proportion of married individuals
mean(data$marital.status == 1)

##############
# Part (d)
##############

# Calculate the proportion of females
mean(data$gender == 0)

##############
# Part (e)
##############

# Total number of observations with a value in any column
sum(!is.na(data))
# Total number of NA values in the dataset
sum(is.na(data))

##############
# Part (f)
##############

# Convert the "income" variable to a factor
data$income <- as.factor(data$income)


#################
# Question (iv)
#################

##############
# Part (a)
##############

# Calculate the number of observations for the training set
train_size <- floor(nrow(data) * 0.70)

# Print the last training observation
cat("The last training set observation is row:",train_size , "\n")

##############
# Part (b)
##############

# Create the training data table
dt_train <- data[1:train_size, ]
head(dt_train)

##############
# Part (c)
##############

# Calculate the size of the test set
test_size <- floor(0.3 * nrow(data))

# Create the testing data table
dt_test <- data[(train_size + 1):nrow(data), ]


#################
# Question (v)
#################

##############
# Part (b)
##############

# Create the feature matrix and outcome variable
X <- model.matrix(income ~ ., data)[, -1]  
y <- data$income

# Define the lambda grid (50 evenly spaced values from 10^5 to 10^-2)
lambda_grid <- 10^seq(5, -2, length = 50)

# Create a trainControl object for 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the lasso regression model
lasso_model <- train(x = X, y = y,method = "glmnet", tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid),ntrControl = train_control)

# Display the best model
lasso_model

##############
# Part (c)
##############

# Extract the row with the highest classification accuracy
best_result <- lasso_model$results[which.max(lasso_model$results$Accuracy), ]

# Retrieve the best lambda and its corresponding accuracy
best_lambda <- best_result$lambda
best_accuracy <- best_result$Accuracy

# Display the results
cat("The highest classification accuracy is:", round(best_accuracy, 4), "\n")
cat("The corresponding value of lambda is:", best_lambda, "\n")

##############
# Part (d)
##############

# Extract the coefficients for the best lambda value
lasso_coef <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)

# Convert the coefficients to a data frame for easier handling
coef_df <- as.data.frame(as.matrix(lasso_coef))
colnames(coef_df) <- "Coefficient"

# Identify variables with coefficients approximately zero
zero_coef <- rownames(coef_df[abs(coef_df$Coefficient) < 1e-5, , drop = FALSE])

# Print variables with approximately zero coefficients
zero_coef

##############
# Part (e)
##############

# Define non-zero coefficient variables by excluding zero_coef variables
non_zero_coef <- setdiff(colnames(data), zero_coef)

# Filter the dataset to include only non-zero coefficient variables and the "income" column
filtered_data <- data[, c(non_zero_coef, "income"), drop = FALSE]

# Create feature matrix and outcome variable
X_filtered <- model.matrix(income ~ ., filtered_data)[, -1] 
y_filtered <- filtered_data$income

# Define the lambda grid
lambda_grid <- 10^seq(5, -2, length = 50)

# Create trainControl for 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the lasso regression model
lasso_model_filtered <- train(x = X_filtered, y = y_filtered,
method = "glmnet", tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid), trControl = train_control)

# Display the best lasso model
lasso_model_filtered

# Train the ridge regression model
ridge_model_filtered <- train(x = X_filtered, y = y_filtered, method = "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid), trControl = train_control)

# Display the best ridge model
ridge_model_filtered

# Extract best lambda and corresponding accuracy for lasso
best_lambda_lasso <- lasso_model_filtered$bestTune$lambda
best_accuracy_lasso <- max(lasso_model_filtered$results$Accuracy)

# Extract best lambda and corresponding accuracy for ridge
best_lambda_ridge <- ridge_model_filtered$bestTune$lambda
best_accuracy_ridge <- max(ridge_model_filtered$results$Accuracy)

# Print the results
cat("Best classification accuracy for Lasso:", round(best_accuracy_lasso, 4), "with lambda:", best_lambda_lasso, "\n")
cat("Best classification accuracy for Ridge:", round(best_accuracy_ridge, 4), "with lambda:", best_lambda_ridge, "\n")

# Determine which model has the higher best accuracy
if (best_accuracy_lasso > best_accuracy_ridge) {
  cat("Lasso regression has the higher best classification accuracy.\n")
} else if (best_accuracy_ridge > best_accuracy_lasso) {
  cat("Ridge regression has the higher best classification accuracy.\n")
} else {
  cat("Both models have the same best classification accuracy.\n")
}

#################
# Question (vi)
#################

##############
# Part (b)
##############

# Define the grid of mtry values (number of variables randomly sampled as candidates at each split)
mtry_grid <- expand.grid(mtry = c(2, 5, 9))

# Initialize a list to store models
models_rf <- list()

# Evaluate random forest models with different numbers of trees
for (ntree in c(100, 200, 300)) {
  # Print current number of trees being evaluated
  cat(paste0(ntree, " trees in the forest.\n"))
  
  # Train random forest model
  rf_model <- train(
    income ~ .,  
    data = data,  
    method = "rf",  
    tuneGrid = mtry_grid,  
    trControl = trainControl(method = "cv", number = 5),  
    ntree = ntree  
  )
  
  # Store the model
  models_rf[[paste0("ntree_", ntree)]] <- rf_model
  
  # Print model summary
  print(rf_model)
  cat("------------------------------------------------------\n")
}

# Extract and combine results from each model
results_rf <- do.call(rbind, lapply(names(models_rf), function(model_name) {
  model <- models_rf[[model_name]]  
  best <- model$results[which.max(model$results$Accuracy), ]  
  cbind(ntree = as.numeric(gsub("ntree_", "", model_name)), best)  
}))

# Convert results to a data frame for easier handling
results_rf <- as.data.frame(results_rf)

# Print summary of results
results_rf

##############
# Part (e)
##############

# Use the best random forest model (100 trees with mtry = 2)
best_rf_model <- models_rf[["ntree_100"]]

# Make predictions on the entire training data
predictions_train <- predict(best_rf_model, newdata = data)

# Create the confusion matrix
conf_matrix <- confusionMatrix(predictions_train, data$income)

# Print the confusion matrix
conf_matrix

# Extract confusion matrix values
confusion_table <- conf_matrix$table
false_positives <- confusion_table[1, 2]  # Predicted "1" but actual is "0"
false_negatives <- confusion_table[2, 1]  # Predicted "0" but actual is "1"

# Print FP and FN
cat("False Positives:", false_positives, "\n")
cat("False Negatives:", false_negatives, "\n")

#################
# Question (v)
#################

# Calculate the size of the training set
train_size <- floor(0.7 * nrow(data))

# Split the data
dt_train <- data[1:train_size, ]
dt_test <- data[(train_size + 1):nrow(data), ]

# Create the feature matrix and outcome variable for training
X_train <- model.matrix(income ~ ., dt_train)[, -1]  
y_train <- dt_train$income

# Create the feature matrix and outcome variable for testing
X_test <- model.matrix(income ~ ., dt_test)[, -1]  
y_test <- dt_test$income

# Define the grid of lambda values
lambda_grid <- 10^seq(10, -2, length = 100)

# Train lasso regression with 10-fold cross-validation
cv_lasso <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1, lambda = lambda_grid, nfolds = 10)

# Train ridge regression with 10-fold cross-validation
cv_ridge <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 0, lambda = lambda_grid, nfolds = 10)

# Extract the best lambda values for lasso and ridge
best_lambda_lasso <- cv_lasso$lambda.min
best_lambda_ridge <- cv_ridge$lambda.min

# Make predictions using the lasso model
pred_probs_lasso <- predict(cv_lasso, s = best_lambda_lasso, newx = X_test, type = "response")
predicted_classes_lasso <- ifelse(pred_probs_lasso > 0.5, 1, 0)
accuracy_lasso <- mean(predicted_classes_lasso == y_test)

# Print the lasso test accuracy
cat("Lasso Test Classification Accuracy:", round(accuracy_lasso, 4), "\n")

# Make predictions using the ridge model
pred_probs_ridge <- predict(cv_ridge, s = best_lambda_ridge, newx = X_test, type = "response")
predicted_classes_ridge <- ifelse(pred_probs_ridge > 0.5, 1, 0)
accuracy_ridge <- mean(predicted_classes_ridge == y_test)

# Print the ridge test accuracy
cat("Ridge Test Classification Accuracy:", round(accuracy_ridge, 4), "\n")




