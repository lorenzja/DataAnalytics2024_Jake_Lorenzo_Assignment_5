library(ggfortify)
library(e1071)
library(class)
library(psych)
library(ggplot2)
library(caret)
library(dplyr)
library(readr)

NYC_Citywide_Annualized_Calendar_Sales_Update_20241126 <- read_csv("NYC_Citywide_Annualized_Calendar_Sales_Update_20241126.csv")


#import NYC citywide data

#create a subset containing data from only one borough (Manhattan)

Manhattan_data <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$BOROUGH == 1, ]

#keep only relevant columns
Manhattan_data <- Manhattan_data[, -c(1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)]

#remove NA values
Manhattan_data <- na.omit(Manhattan_data)

#remove zero values
Manhattan_data <- Manhattan_data[rowSums(Manhattan_data == 0) == 0, ]

#removal of "-0" values from the dataset
Manhattan_data <- Manhattan_data[!apply(Manhattan_data == "- 0", 1, any), ]

# Convert all columns except columns  to factors
Manhattan_data[-c(4, 5, 6, 7, 8)] <- lapply(Manhattan_data[-c(4, 5, 6, 7, 8)], as.factor)

# Convert columns 6 and 7 to numeric by removing commas
Manhattan_data[, c(5, 6)] <- lapply(Manhattan_data[, c(5, 6)], function(x) {
  as.numeric(gsub(",", "", x))
})

#outlier removal in SALE PRICE column - remove rows with values < 50000 in sale price column
Manhattan_data <- Manhattan_data[Manhattan_data$`SALE PRICE` >= 50000, ]

# Outlier removal in SALE PRICE column - remove rows with values > 100,000,000
Manhattan_data <- Manhattan_data[Manhattan_data$`SALE PRICE` <= 100000000, ]

# Remove rows where GROSS SQUARE FEET is greater than 50,000
Manhattan_data <- Manhattan_data[Manhattan_data$`GROSS SQUARE FEET` <= 50000, ]

# Data for x is in columns 2-8
x <- Manhattan_data[, c(2:7)]

# y is in column 8
y <- Manhattan_data[, 8]

## feature boxplots
boxplot(x, main="manhattan features")

## class label distributions
plot(y, main="manhattan sale prices")

# Combine x and y into one dataframe for plotting
data_combined <- data.frame(x, Class = y)

# Create the Pairs Panels Plot
pairs.panels(data_combined,
             gap = 0,
             bg = c("red", "blue")[data_combined$Class],
             pch = 21,  # Point shape
             hist.col = "lightblue",  # Histogram color
             main = "Pairs Panels Plot")

# Create the ggplot
options(scipen = 999)  # Disable scientific notation
ggplot(Manhattan_data, aes(x = `GROSS SQUARE FEET`, y = `SALE PRICE`, colour = `SALE PRICE`)) +
  geom_point() +
  labs(title = "Sale Price vs Gross sq feet",
       x = "Gross sq feet",
       y = "Sale Price",
       colour = "Sale Price") +
  theme_minimal()

# Convert columns to log base 10 form
Manhattan_data$`LAND SQUARE FEET` <- log10(Manhattan_data$`LAND SQUARE FEET`)
Manhattan_data$`GROSS SQUARE FEET` <- log10(Manhattan_data$`GROSS SQUARE FEET`)
Manhattan_data$`SALE PRICE` <- log10(Manhattan_data$`SALE PRICE`)


# Split the dataset into training and testing sets
set.seed(123)  # For reproducibility
train.indexes <- sample(nrow(Manhattan_data), 0.7 * nrow(Manhattan_data))

# Create training and testing datasets
train <- Manhattan_data[train.indexes, ]
test <- Manhattan_data[-train.indexes, ]

# Train a linear model to predict SALE PRICE based on LAND SQUARE FEET, GROSS SQUARE FEET, and YEAR BUILT
linear_model_NY <- lm(`SALE PRICE` ~ 
                        `TOTAL UNITS` + 
                        `LAND SQUARE FEET` + 
                        `GROSS SQUARE FEET` + 
                        `YEAR BUILT`, 
                      data = train)

# Summary of the linear model
summary(linear_model_NY)

#plot linear model
plot(linear_model_NY)

# Predict on the testing data
test$Predicted_PRICE <- predict(linear_model_NY, newdata = test)

# Plot predicted price vs. actual price for the testing dataset
ggplot(test, aes(x = `SALE PRICE`, y = Predicted_PRICE)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Predicted vs. Actual Prices (Linear Model)",
    x = "Actual Sale Price (log10)",
    y = "Predicted Sale Price (log10)"
  ) +
  theme_minimal()

##############################Part 1 Part D - Classification (SVM and KNN)

## train SVM model - linear kernel
svm.mod0 <- svm(`LOT` ~ ., data = train, kernel = 'linear')

svm.mod0

train.pred <- predict(svm.mod0, train)

cm = as.matrix(table(Actual = train$LOT, Predicted = train.pred))

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted 

recall = diag / rowsums 
precision = diag / colsums
f1 = 2 * precision * recall / (precision + recall) 

data.frame(precision, recall, f1)

accuracy <- sum(diag(cm)) / sum(cm)
print(paste("Accuracy:", accuracy))


# Plot predicted vs. actual values
ggplot(data.frame(Actual = test$`LOT`, Predicted = test.pred), aes(x = Actual, y = Predicted)) +
  geom_point(color = "red") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Predicted vs. Actual Prices", x = "Actual Prices", y = "Predicted Prices")


#KNN model for same as above
# Normalize numeric columns (replace 3:8 with actual numeric columns indices if needed)
normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }

# Apply normalization to selected columns (adjust column indices as necessary)
Manhattan_data[4:8] <- as.data.frame(lapply(Manhattan_data[4:8], normalize))

# Summary of the filtered data
summary(Manhattan_data)

# Create a sample for splitting (adjust the sample size based on your data)
set.seed(123)
s_Manhattan <- sample(nrow(Manhattan_data), size = floor(0.7 * nrow(Manhattan_data)))

# Split into train and test datasets
Manhattan.train <- Manhattan_data[s_Manhattan, ]
Manhattan.test <- Manhattan_data[-s_Manhattan, ]

sqrt(4483)

# Set k (e.g., sqrt of train size)
k <- 67

# Perform KNN classification with updated indices
KNNpred <- knn(train = Manhattan.train[c(1, 3:8)], 
               test = Manhattan.test[c(1, 3:8)], 
               cl = Manhattan.train$LOT, 
               k = k)

# Create a contingency table to evaluate predictions
contingency.table <- table(KNNpred, Manhattan.test$LOT)
contingency.table

# Convert to matrix to calculate accuracy
contingency.matrix <- as.matrix(contingency.table)
accuracy <- sum(diag(contingency.matrix)) / sum(contingency.matrix)
print(paste("Accuracy:", accuracy))

# Initialize the accuracy vector
accuracy <- c()

# Set the k values to test
ks <- c(61, 63, 65, 67, 69)

# Loop over different k values
for (k in ks) {
  # Perform KNN classification for each k
  KNNpred <- knn(train = Manhattan.train[c(1, 3:8)], 
                 test = Manhattan.test[c(1, 3:8)], 
                 cl = Manhattan.train$LOT, 
                 k = k)
  
  # Create a contingency table
  contingency.table <- table(KNNpred, Manhattan.test$LOT)
  contingency.matrix = as.matrix(contingency.table) 
  
  # Calculate accuracy for this k
  accuracy_k <- sum(diag(contingency.matrix)) / sum(contingency.matrix)
  
  # Append accuracy to the accuracy vector
  accuracy <- c(accuracy, accuracy_k)
}

# Plot accuracy vs k values
plot(ks, accuracy, type = "b", 
     xlab = "k (Number of Neighbors)", ylab = "Accuracy", 
     main = "KNN Accuracy for Different k Values")

########################################################################
############################Part 2######################################
########################################################################
# Keep only relevant columns
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126 <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[, -c(1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)]

# Remove NA values
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126 <- na.omit(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126)

# Remove zero values
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126 <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[rowSums(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126 == 0) == 0, ]

# Removal of "-0" values from the dataset
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126 <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[!apply(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126 == "- 0", 1, any), ]

# Convert all columns except columns 4, 5, 6, 7, and 8 to factors
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[-c(4, 5, 6, 7, 8)] <- lapply(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[-c(4, 5, 6, 7, 8)], as.factor)

# Convert columns 5 and 6 to numeric by removing commas
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[, c(5, 6)] <- lapply(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[, c(5, 6)], function(x) {
  as.numeric(gsub(",", "", x))
})

# Outlier removal in SALE PRICE column - remove rows with values < 50000 in sale price column
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126 <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$`SALE PRICE` >= 50000, ]

# Outlier removal in SALE PRICE column - remove rows with values > 750,000,000
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126 <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$`SALE PRICE` <= 75000000, ]

# Remove rows where GROSS SQUARE FEET is greater than 50,000
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126 <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$`GROSS SQUARE FEET` <= 50000, ]

# Data for x is in columns 2-8
x <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[, c(2:7)]

# y is in column 8 (SALE PRICE)
y <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$`SALE PRICE`

## Feature boxplots
boxplot(x, main="NYC Citywide Features")

## Class label distributions (SALE PRICE)
plot(y, main="NYC Citywide Sale Prices")

# Create the ggplot
options(scipen = 999)  # Disable scientific notation
ggplot(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126, aes(x = `GROSS SQUARE FEET`, y = `SALE PRICE`, colour = `SALE PRICE`)) +
  geom_point() +
  labs(title = "Sale Price vs Gross Square Feet",
       x = "Gross Square Feet",
       y = "Sale Price",
       colour = "Sale Price") +
  theme_minimal()

# Convert columns to log base 10 form
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$`LAND SQUARE FEET` <- log10(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$`LAND SQUARE FEET`)
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$`GROSS SQUARE FEET` <- log10(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$`GROSS SQUARE FEET`)
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$`SALE PRICE` <- log10(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126$`SALE PRICE`)

# Split the dataset into training and testing sets
set.seed(123)  # For reproducibility
train.indexes <- sample(nrow(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126), 0.7 * nrow(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126))

# Create training and testing datasets
train <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[train.indexes, ]
test <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[-train.indexes, ]

# Train a linear model to predict SALE PRICE based on TOTAL UNITS, LAND SQUARE FEET, GROSS SQUARE FEET, and YEAR BUILT
linear_model_NYC <- lm(`SALE PRICE` ~ 
                         `TOTAL UNITS` + 
                         `LAND SQUARE FEET` + 
                         `GROSS SQUARE FEET` + 
                         `YEAR BUILT`, 
                       data = train)

# Summary of the linear model
summary(linear_model_NYC)

# Plot linear model
plot(linear_model_NYC)

# Predict on the testing data
test$Predicted_PRICE <- predict(linear_model_NYC, newdata = test)

# Plot predicted price vs. actual price for the testing dataset
ggplot(test, aes(x = `SALE PRICE`, y = Predicted_PRICE)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Predicted vs. Actual Prices (Linear Model)",
    x = "Actual Sale Price (log10)",
    y = "Predicted Sale Price (log10)"
  ) +
  theme_minimal()

####PART B#####
# Normalize numeric columns (replace 4:8 with the actual numeric column indices if necessary)
normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }

# Apply normalization to selected columns in the NYC dataset
NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[4:8] <- as.data.frame(
  lapply(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[4:8], normalize)
)

# Summary of the normalized data
summary(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126)

# Create a sample for splitting (70% training, 30% testing)
set.seed(123)
sample_indices <- sample(nrow(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126), 
                         size = floor(0.7 * nrow(NYC_Citywide_Annualized_Calendar_Sales_Update_20241126)))

# Split into train and test datasets
train_data <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[sample_indices, ]
test_data <- NYC_Citywide_Annualized_Calendar_Sales_Update_20241126[-sample_indices, ]

sqrt(136783)

# Set k to a value close to sqrt of the training dataset size
k <- 3

# Perform KNN classification
KNNpred <- knn(
  train = train_data[c(1, 3:8)], 
  test = test_data[c(1, 3:8)], 
  cl = train_data$LOT, 
  k = k
)

# Create a contingency table to evaluate predictions
contingency_table <- table(KNNpred, test_data$LOT)
print(contingency_table)

# Convert the table to a matrix and calculate accuracy
contingency_matrix <- as.matrix(contingency_table)
accuracy <- sum(diag(contingency_matrix)) / sum(contingency_matrix)
print(paste("Accuracy:", accuracy))

# Initialize an accuracy vector for different k values
accuracy_vector <- c()

# Define the k values to test
k_values <- c(3, 7, 11, 13, 19)

# Loop through each k value to evaluate performance
for (k in k_values) {
  KNNpred <- knn(
    train = train_data[c(1, 3:8)], 
    test = test_data[c(1, 3:8)], 
    cl = train_data$LOT, 
    k = k
  )
  
  # Create a contingency table and calculate accuracy for this k
  contingency_table <- table(KNNpred, test_data$LOT)
  contingency_matrix <- as.matrix(contingency_table)
  accuracy_k <- sum(diag(contingency_matrix)) / sum(contingency_matrix)
  
  # Append accuracy to the vector
  accuracy_vector <- c(accuracy_vector, accuracy_k)
}

# Plot accuracy vs k values
plot(
  k_values, accuracy_vector, type = "b", 
  xlab = "k (Number of Neighbors)", ylab = "Accuracy", 
  main = "KNN Accuracy for Different k Values"
)



