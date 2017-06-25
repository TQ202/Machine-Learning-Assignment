# Predicting Weightlifting Class Using Machine Learning Techniques





## Background Information

Six individuals were asked to perform one set of ten repetitions of unilateral dumbbell bicep curls correctly and incorrectly in the following ways: 

* **Class A** - exactly according to the specification
* **Class B** - throwing the elbows to the front
* **Class C** - lifting the dumbbell only halfway
* **Class D** - lowering the dumbbell only halfway
* **Class E** - throwing the hips to the front

Class A corresponds to the specified execution of the exercise, while the other four classes correspond to common mistakes. Each participant wore accelerometers on the belt, forearm, arm, as well as the dumbell to measure the movements associated with each lift. The goal of this analysis is to predict the manner in which each participant did the exercise using any of the variables that the accelerometer measured.

## Out-of-Sample Error Prediction

Out-of-sample error measures the accuracy of the model on the testing data set. The expectation is that the out-of-sample error will be greater than the in-sample error due to overfitting of the model. 

I expect the out-of-sample error to be quite small due to following factors:

* Large data sample used to train model (n = 19,622)
* 160 predictor variables to choose from to build model
* Random Forest and Boosting are two of the most accurate modeling choices

## Load Data

Loading of the data:


```r
training_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(training_url, "training.csv")
training <- read.csv("training.csv", na.strings = c("NA", "", "#DIV/0!"))

testing_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(testing_url, "testing.csv")
testing <- read.csv("testing.csv", na.strings = c("NA", "", "#DIV/0!"))
```

## Load Packages

Load packages used for data analysis:


```r
library(caret)
library(randomForest)
library(parallel)
library(doParallel)
```

## Set Seed

In order to make the data reproducible, it is necessary to set a seed value:


```r
set.seed(5678)
```

## Cleaning Data

Find the variables with near zero variance and eliminate them from the training and testing sets:


```r
NZV <- nearZeroVar(training)
training_NZV <- training[, -NZV]
testing_NZV <- testing[, -NZV]
```

Eliminate variables with too many "NA" values (90+ percent of their total number of values) from the training and testing sets:


```r
NA_percentage <- function(x) {sum(is.na(x))/length(x)}
training_NA <- sapply(training_NZV[, 1:ncol(training_NZV)], NA_percentage)
training_NA_index <- training_NA < 0.9
training_NA <- training_NZV[, training_NA_index]
testing_NA <- testing_NZV[, training_NA_index]
```

Eliminate variables that should not be a part of the final model:


```r
training_final <- training_NA[, !names(training_NA) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "num_window")]

testing_final <- testing_NA[, !names(testing_NA) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "num_window")]
```

## Cross Validation

Split the training set into a new training (60% of the original training set) and testing set (40% of the original training set). This allows the model to be tested without having to use the final testing set. Once the most accurate model is chosen, the model can be tested on the untouched final testing set:


```r
inTrain <- createDataPartition(y = training_final$classe, p = 0.6, list = FALSE)
training_set <- training_final[inTrain, ]     
testing_set <- training_final[-inTrain, ]    
```

## Boosted Model

Fit a boosted model predictor relating the variable "classe" to the remaining variables and then predict the "classe" based on the testing set: 


```r
fit_gbm <- train(classe ~ ., method = "gbm", data = training_set, verbose = FALSE)
pred_gbm <- predict(fit_gbm, newdata = testing_set)
```

Set up a confusion matrix to see what the out-of-sample error looks like:


```r
confusionMatrix(pred_gbm, testing_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2195   39    0    2    1
##          B   18 1431   38    8   13
##          C   15   43 1314   50    9
##          D    1    3   13 1220   25
##          E    3    2    3    6 1394
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9628          
##                  95% CI : (0.9584, 0.9669)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9529          
##  Mcnemar's Test P-Value : 2.725e-11       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9834   0.9427   0.9605   0.9487   0.9667
## Specificity            0.9925   0.9878   0.9819   0.9936   0.9978
## Pos Pred Value         0.9812   0.9489   0.9182   0.9667   0.9901
## Neg Pred Value         0.9934   0.9863   0.9916   0.9900   0.9925
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2798   0.1824   0.1675   0.1555   0.1777
## Detection Prevalence   0.2851   0.1922   0.1824   0.1608   0.1795
## Balanced Accuracy      0.9880   0.9653   0.9712   0.9711   0.9823
```

The boosted model was able to predict the correct weightlifting class in the test set with 96.28% accuracy (out-of-sample error = 3.72%).

## Random Forest Model:

Fit a random forest model predictor relating the variable "classe" to the remaining variables and then predict the "classe" based on the testing set. In order to reduce the run time, parallel processing is used and the trainControl function is configured:


```r
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

fit_rf <- train(classe ~ ., method = "rf", data = training_set, tr_contrl = fitControl)

stopCluster(cluster)
registerDoSEQ()

pred_rf <- predict(fit_rf, newdata = testing_set)
```

Set up a confusion matrix to see what the out-of-sample error looks like:


```r
confusionMatrix(pred_rf, testing_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230   10    0    0    0
##          B    0 1504   15    0    0
##          C    2    4 1352   32    2
##          D    0    0    1 1254    7
##          E    0    0    0    0 1433
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9907          
##                  95% CI : (0.9883, 0.9927)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9882          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9908   0.9883   0.9751   0.9938
## Specificity            0.9982   0.9976   0.9938   0.9988   1.0000
## Pos Pred Value         0.9955   0.9901   0.9713   0.9937   1.0000
## Neg Pred Value         0.9996   0.9978   0.9975   0.9951   0.9986
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1917   0.1723   0.1598   0.1826
## Detection Prevalence   0.2855   0.1936   0.1774   0.1608   0.1826
## Balanced Accuracy      0.9987   0.9942   0.9911   0.9869   0.9969
```

The random forest model was able to predict the correct weightlifting class in the test set with 99.07% accuracy (out-of-sample error = 0.93%).

## Conclusion

While the boosted model performed well, the best results came from the random forest model. Thus, the random forest model will be used for the test case prediction.

## Test Case Prediction

Using the random forest model fit, predict the "classe" for the final testing set:


```r
pred_final <- predict(fit_rf, newdata = testing_final[,-58])
```

## Course Project Prediction Quiz Portion

Store the results of the prediction in an individual text file:


```r
for (i in 1:length(pred_final)) {
    
    write.table(pred_final[i], 
                file = paste0("problem_number", i, ".txt"), 
                quote = FALSE, 
                row.names = FALSE, 
                col.names = FALSE)
}
```
