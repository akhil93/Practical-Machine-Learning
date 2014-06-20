Practical Machine Learning.
========================================================

# Introduction.

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, i will be using  data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

# Exloratory Data Analysis.
Lets begin our analysis by loading our data set.

```r
train.raw = read.csv("pml-training.csv")
test.raw = read.csv("pml-testing.csv")
```

As we can see our training data set is very huge, lets divide that dataset for validation purpose. For Data Splitting i have used the 'sample.split' function from the 'caTools' package.the function Split data from vector Y into two sets in predefined ratio while preserving relative ratios of different labels in Y. Used to split the data used during classification into train and test subsets.

```r
library(caTools)
```

```
## Warning: package 'caTools' was built under R version 3.0.3
```

```r
split = sample.split(train.raw, SplitRatio = 0.7)
# we are telling to split the data into a 70% chunk and remaining as another
# chunk.
TRAIN = subset(train.raw, split == TRUE)  # 70% of the data.
TEST = subset(train.raw, split == FALSE)  # 30% of the data.
```


Upon inspecting the data we find there are a lot of missing(NA) values and other non- computable values, more over we have a lot of features in the data. We need to select only some of them. The approach i followed is calculating the 'nearZeroVar'. It is a function from the 'caret' package. nearZeroVar diagnoses predictors that have one unique value  or predictors that are have both of the following characteristics: they have very few unique values relative to the number of samples and the ratio of the frequency of the most common value to the frequency of the second most common value is large. Coming to the type of the features, i choose to have
variables of type 'numeric'.

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.0.3
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.0.3
```

```
## Loading required package: ggplot2
```

```r
NZV = nearZeroVar(TRAIN)
TRAIN = TRAIN[-NZV]
TEST = TEST[-NZV]
test.raw = test.raw[-NZV]
imp.features = which(lapply(TRAIN, class) %in% c("numeric"))
```


The next thing is to impute the data and process it for applying different prediction models.
To process the data, i have used the 'preProcess()' from the "caret" package. Why preProcess() because the transformation can be estimated from the training data and applied to any data set with the same variables.

```r
prePro = preProcess(TRAIN[, imp.features], method = "knnImpute")
p.train = cbind(TRAIN$classe, predict(prePro, TRAIN[, imp.features]))  # cleaned train dataset.
names(p.train)[1] = "classe"
p.test = cbind(TEST$classe, predict(prePro, TEST[, imp.features]))  # cleaned test dataset.
names(p.test)[1] = "classe"
p.outTest = predict(prePro, test.raw[, imp.features])  # cleaned out-sample test dataset. 
```


# Prediction Models.

Lets try "randomForest" model because it is one of the most widely used prediction algorithm because it is more accurate than others.


```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.0.3
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rf_model = randomForest(classe ~ ., data = p.train, ntree = 1000, mtry = 32)
rf_model
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = p.train, ntree = 1000,      mtry = 32) 
##                Type of random forest: classification
##                      Number of trees: 1000
## No. of variables tried at each split: 32
## 
##         OOB estimate of  error rate: 0.91%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3885   11    3    0    6    0.005122
## B   17 2629   13    0    1    0.011654
## C    2   13 2358   17    5    0.015449
## D    1    0   14 2232    5    0.008881
## E    1    4    5    7 2505    0.006741
```

```r
# ntree is set 1000 because we want to make sure every input gets predicted
# few times and at the end the model can choose the optimal one.
```

Lets try to predict on the in-sample and out -sample test data.

```r
training_pred = predict(rf_model, p.train)
table(training_pred, p.train$classe)
```

```
##              
## training_pred    A    B    C    D    E
##             A 3905    0    0    0    0
##             B    0 2660    0    0    0
##             C    0    0 2395    0    0
##             D    0    0    0 2252    0
##             E    0    0    0    0 2522
```

```r
# to calculate the accuracy we sum the diagonal cells and divide them by the
# total number of rows. But here we can see that the model has predicted
# with 100% accuracy.

# For more detailed summary use the confusionMatrix() instead of table()

testing_pred = predict(rf_model, p.test)
table(testing_pred, p.test$classe)
```

```
##             
## testing_pred    A    B    C    D    E
##            A 1671   11    0    0    2
##            B    3 1122    9    1    0
##            C    0    4 1011    6    3
##            D    1    0    6  955    3
##            E    0    0    1    2 1077
```

```r

(1668 + 1120 + 1004 + 953 + 1076)/5888  # out sample accuracy.
```

```
## [1] 0.9886
```


# Results.

We calculated the in sample and out sample accuracy of our model and we are very much satisfied by its performance, so lets try to predict the main test data which contains 20 cases.

```r
answers <- predict(rf_model, p.outTest)
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

The following code is provided by the course staff to create 20 different .txt files where each file contains the predicted classe value of respective test case.

```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(answers)
```


In the final test data with 20 test cases the above mentioned model has accurately predicted all the classe type with 100 % accuracy.

# Reference.
[1] - Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 
Cited by 2 (Google Scholar)



