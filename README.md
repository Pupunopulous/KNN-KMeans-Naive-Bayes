# Implementing KNN, KMeans and Naive-Bayes algorithms

### Author: 
Rahi Krishna / rk4748

## Running the Program

The `.java` file can be executed on any operating system with JDK 8.0 or higher. Follow the steps below:

###Compile the Java file:

```shell
javac learn.java
java learn {arg1 arg2 arg3 ...}
```
#### The following arguments are accepted (if appended after `java Minimax`):
IMPORTANT: Order of these arguments does not matter

#### Running KNN:
Required command line arguments:
1. `-train $fileName$` - An input file for KNN to train the algorithm
2. `-test $fileName$` - An input file for KNN to test the algorithm
3. `-k $val$` - A value indicating the number of neighbors to train the algorithm on
If `k = 0`, the program performs Naive-Bayes instead
Input files must be of the format `$input$.txt`

#### Optional command line arguments:
1. `-v` or `-verbose` - Gives a verbose output showing the test procedure


#### Running Naive-Bayes:
Required command line arguments:
1. `-train $fileName$` - An input file for Naive-Bayes to train the algorithm
2. `-test $fileName$` - An input file Naive-Bayes to test the algorithm
3. `-c $val$` - A value indicating the Laplacian correction to train the algorithm on
If not provided, the program defaults to a Laplacian correction of 0: `-c 0`
Input files must be of the format `$input$.txt`

#### Optional command line arguments:
1. `-v` or `-verbose` - Gives a verbose output showing the test procedure


#### Running K-means:
#### Required command line arguments:
1. `-train $fileName$` - An input file for K-means to train the algorithm and find the final clusters
2. `-d $manh$` or `-d $e2$` - A value indicating whether to use manhattan (manh) or euclidean distance (e2) as the distance comparator
3. `x1,y1  x2,y2  x3,y3 ...` or `x1,y1,z1  x2,y2,z2  x3,y3,z3 ...` comma separated values as the initial centroids for running K-means
Centroid dimensions must be the same as the input file nodes
Input files must be of the format `$input$.txt`


#### Some example commands:
```shell
javac learn.java // Required

// Runs KNN
java learn -train train.txt -test test.txt -k 3
java learn -test test.txt -train train.txt -verbose -k 4

// Runs Naive-Bayes
java learn -train train.txt -test test.txt -c 1 -v

// Runs K-means
java learn -train input.txt 0,0 200,200 500,500 -d manh
java learn 0,0 10,10 100,100 -train input.txt -d e2
```

#### IMPORTANT:
Please make sure `KNN.java`, `NaiveBayes.java`, `KMeans.java`, `Evaluator.java` and `learn.java` are in the same folder, along with all the input files
