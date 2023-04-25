# EE 399 HW 3
### Zack Martin

## Abstract
This project investigates the recognition and classification of handwritten digits using various machine learning techniques such as correlation matrices, singular value decomposition (SVD), linear discriminant analysis (LDA), support vector machines (SVM), and decision trees. We explore the well-known MNIST dataset and preprocess the images into a suitable format for analysis. By computing correlation matrices, we analyze the similarities and differences between digits and uncover the underlying structure of the data through eigenvectors and SVD. Our investigation of the principal component directions emphasizes the importance of these techniques in dimensionality reduction and feature extraction for applications in computer vision and machine learning.
We then compare the performance of LDA, SVM, and decision tree classifiers in separating and identifying individual digits. Through this analysis, we gain insights into the strengths and weaknesses of these methods for digit recognition tasks. We also examine the trade-offs between classification accuracy and computational efficiency and discuss the implications of these results for real-world applications of machine learning. Finally, we provide recommendations for future research to improve the accuracy and scalability of digit recognition algorithms and identify areas where machine learning can be applied to other related problems in computer vision and pattern recognition.

## Introduction and Overview
This project focuses on the classification of handwritten digits from the MNIST dataset using various machine-learning techniques such as singular value decomposition (SVD), linear discriminant analysis (LDA), support vector machines (SVM), and decision trees. We begin by applying SVD to reduce the dimensionality of the data and create a feature space that serves as input for the classification models. Using the entire dataset, we train and test LDA, SVM, and decision tree classifiers, analyzing their performance on all ten digits.
In addition to the overall analysis, we investigate which digits are the most difficult to separate and the easiest to separate. We use LDA as a base case to train a two-digit classifier to separate any two/three digits. From this model, we determine the hardest and easiest pairs of digits to separate and proceed to train and test SVM and decision tree classifiers on these pairs. We compare the results and provide insights into the strengths and weaknesses of each classification method for the most challenging and easiest digit classification tasks.
Furthermore, we explore the concept of correlation matrices and eigenvectors to uncover underlying structures and relationships within the data. By analyzing the linear relationship between pairs of images in our dataset, we gain insights into the similarities and differences between digits. We use eigenvectors and eigenvalues to identify key patterns in the data and highlight their significance in understanding the structure of handwritten digits.
Overall, this project provides a comprehensive analysis of the classification of handwritten digits using various machine-learning techniques and offers insights into the strengths and limitations of each method. The investigation of correlation matrices and eigenvectors further enhances our understanding of the underlying structure of handwritten digits.

## Theoretical Background

### Correlation Matrices

A correlation matrix is a powerful tool in data analysis that captures the linear relationships between pairs of variables. It is a square matrix that contains the Pearson correlation coefficients between the variables, which are computed as the dot product between the standardized variables. The resulting matrix shows how each variable is related to every other variable in the dataset.

In our analysis, we calculate the correlation between pairs of images by computing the dot product between the image matrices.

### Eigenvectors and Eigenvalues

Eigenvectors and eigenvalues are important concepts in linear algebra that are used to analyze the structure of matrices. For a given square matrix A, an eigenvector v and its corresponding eigenvalue λ satisfy the following equation:

$$A\textbf{v} = \lambda\textbf{v}$$

Eigenvectors represent the directions in which the matrix A stretches or compresses the data, while eigenvalues indicate the magnitude of that stretching or compression. In our analysis, we compute the eigenvectors and eigenvalues of the correlation matrix to identify key patterns in the handwritten digit dataset.

### Singular Value Decomposition (SVD)

Singular value decomposition (SVD) is a factorization technique that decomposes a given matrix X into three matrices U, S, and V^T:

$$X = USV^T$$

U and V are orthogonal matrices containing the left and right singular vectors of X, respectively, while S is a diagonal matrix containing the singular values of X. SVD can be used for dimensionality reduction, feature extraction, and data compression by selecting a subset of singular vectors that capture the most significant variations in the data.

The SVD is calculated as follows:

1. Calculate the correlation matrix $\textbf{C} = \textbf{X}^T\textbf{X}$
2. Compute the eigenvectors and eigenvalues of $\textbf{C}$
3. Construct the matrix $\textbf{U}$ from the eigenvectors of $\textbf{C}$
4. Construct the diagonal matrix $\textbf{S}$ from the square roots of the eigenvalues of $\textbf{C}$
5. Compute the matrix $\textbf{V}$ using the formula $\textbf{V} = \frac{1}{\sqrt{n-1}}\textbf{X}\textbf{U}\textbf{S}^{-1}$

In our analysis, we perform SVD on the handwritten digit matrix $\textbf{X}$ and examine the principal component directions to uncover the underlying structure of the data.

### Linear Discriminant Analysis (LDA)

Linear discriminant analysis (LDA) is a supervised learning technique that is used for dimensionality reduction and classification. LDA involves finding a linear combination of features that maximally separates the classes in the dataset.

The LDA is calculated as follows:

1. Calculate the mean vectors $\textbf{m}_i$ for each class in the dataset
2. Calculate the within-class scatter matrix $\textbf{S}_w$ and between-class scatter matrix $\textbf{S}_b$
3. Compute the eigenvectors

## Algorithm Implementation and Development
We begin by loading the dataset and reshaping the data.

```
mnist = fetch_openml('mnist_784')
X, Y = mnist['data'].to_numpy(), mnist['target']

X_flat = X.reshape(X.shape[0], -1)
scaler = StandardScaler()
X_stand = scaler.fit_transform(X_flat)
```

### Part 1
In this part, we perform an SVD analysis on the reshaped images.

```
U, s, Vt = np.linalg.svd(X_stand, full_matrices=False)
```

### Part 2
For part 2, we’ll examine the singular value spectrum through plotting, then determine the necessary modes (rank r) of the digit space.

```
# Calculate the cumulative sum of singular values
cumulative_sum = np.cumsum(svd.singular_values_)


# Plot the cumulative sum of singular values
plt.plot(cumulative_sum / cumulative_sum[-1])
plt.xlabel('Number of Singular Values')
plt.ylabel('Cumulative Sum')
plt.show()


# Find the number of modes needed for 95% variance retention
num_modes = np.argmax(cumulative_sum >= 0.95*cumulative_sum[-1]) + 1
print(f"Number of modes needed for 95% variance retention: {num_modes}")
```

### Part 4
In part 4, we project three V-modes onto a 3D plot. The code for this is as follows:

```
# three selected v-modes
V = svd.components_.T
v_cols = [2, 3, 5]


# v modes mm with data
X_proj = X_stand @ Vt[([mode - 1 for mode in v_cols]), :].T


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for digit in range(10):
    indices = (Y == str(digit))
    ax.scatter(X_proj[indices, 0], X_proj[indices, 1], X_proj[indices, 2], label=f'Digit {digit}')


ax.set_xlabel(f'V-mode {v_cols[0]}')
ax.set_ylabel(f'V-mode {v_cols[1]}')
ax.set_zlabel(f'V-mode {v_cols[2]}')
ax.legend()
plt.show()
```

### 2 - Digit LDA
In this part, we’ll construct a linear classifier (LDA)  for two digits. In our case, those digits are “3” and “8”. 

```
# select 2 digits
selected_digits = ['3', '8']
mask = np.isin(Y, selected_digits)
X_filt, y_filt = X[mask], Y[mask]
X_flat = X_filt.reshape(X_filt.shape[0], -1)
scaler = StandardScaler()
X_stand = scaler.fit_transform(X_flat)

# split into training/ test
X_train, X_test, y_train, y_test = train_test_split(X_stand, y_filt, test_size=0.2, random_state=42)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# check against test data
y_pred = lda.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

accuracy_3 = cm[0, 0] / cm[0].sum()
accuracy_8 = cm[1, 1] / cm[1].sum()
```

### 3 - Digit LDA
Next, we perform the same LDA on a 3-digit sample.

```
# select three digits
selected_digits = ['5', '7', '9']
mask = np.isin(Y, selected_digits)
X_filt, y_filt = X[mask], Y[mask]
X_flat = X_filt.reshape(X_filt.shape[0], -1)
scaler = StandardScaler()
X_stand = scaler.fit_transform(X_flat)

# split into training/ test
X_train, X_test, y_train, y_test = train_test_split(X_stand, y_filt, test_size=0.2, random_state=42)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = lda.predict(X_test)

# check against test data
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

accuracy_5 = cm[0, 0] / cm[0].sum()
accuracy_7 = cm[1, 1] / cm[1].sum()
accuracy_9 = cm[2, 2] / cm[2].sum()
```

### Most Correlated Digits
In order to determine which digits in the data set are most easy to separate, we quantify the accuracy of the separation of each combination of digits then find the maximum accuracy.

```
digit_pairs = list(itertools.combinations(range(10), 2))
accuracy = []

# compute each pair's accuracy
for pair in digit_pairs:
    selected_digits = [str(d) for d in pair]
    mask = np.isin(Y, selected_digits)
    X_filt, y_filt = X[mask], Y[mask]

    X_flat = X_filt.reshape(X_filt.shape[0], -1)
    scaler = StandardScaler()
    X_stand = scaler.fit_transform(X_flat)

    X_train, X_test, y_train, y_test = train_test_split(X_stand, y_filt, test_size=0.2, random_state=42)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    y_pred = lda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    accuracy_1 = cm[0, 0] / cm[0].sum()
    accuracy_2 = cm[1, 1] / cm[1].sum()
    avg_accuracy = (accuracy_1 + accuracy_2) / 2

    accuracy.append(avg_accuracy)
    print(f"Accuracy for digit pair {pair}: {avg_accuracy:.2f}")

min_index = np.argmin(accuracy)
max_index = np.argmax(accuracy)
```

### SVM and Decision Tree Classifiers
Next we want to see how well SVM and decision trees will perform on the digit separation problem. 

As we did in HW 2, we can perform an SVM on the data.

```
# SVM
svm = SVC(kernel='rbf', C=1, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
```

Using the same sklearn library, we can create a decision tree classifier.

```
# Decision Tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
accuracy_dtree = accuracy_score(y_test, y_pred_dtree)
```

### Compare LDA vs SVM vs Decision Tree
Finally, we’ll compare the three models we’ve implemented to see how they stack up against eachother.

With a simple module, we can assess the accuracy of each classifier.
```
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = [cm[i, i] / cm[i].sum() for i in range(cm.shape[0])]
    avg_accuracy = np.mean(accuracy)
    return avg_accuracy
```

## Computational Results

### Part 1
In part 1, we determined the most significant SVD modes and I plotted them to compare.
![download](https://user-images.githubusercontent.com/129991497/234185487-5be66412-4905-4764-96ad-694ca110c54d.png "Fig 1. First 9 most significant SVD Modes")

### Part 2
For part 2, we looked at the singular value spectrum and assesed the digit space to determine how many modes are necessary for good image reconstruction. I plotted the singular value spectrum to get a rough idea vidually of what it looked like.
![download](https://user-images.githubusercontent.com/129991497/234185918-670a428c-e453-4e06-ae94-d234659625ac.png "Fig 2. Singular Value Spectrum")

I then plotted the cumulative singular values and calculated the number of modes needed for a reasonably lov variance.
![download](https://user-images.githubusercontent.com/129991497/234186130-d6a8ded3-a240-4f2a-bb57-1ca692621597.png "Fig 3. Cumulative Singular Values")

In my case, I chose a reasonable variance to be 5% so I calculated the number of modes to satisfy a 95% variance retention to be "427"

### Part 3
In part 3, we're asked Wwat the interpretation of the U, Σ, and V matrices are.

In the SVD decomposition, the input data matrix X is decomposed into three matrices:

U: an m x m orthogonal matrix whose columns represent the left singular vectors of X.
Σ: an m x n diagonal matrix whose entries represent the singular values of X.
V: an n x n orthogonal matrix whose columns represent the right singular vectors of X.
The interpretation of these matrices is as follows:

U: The columns of U are the eigenvectors of X X^T (or 1/n X X^T if we use normalized data), which correspond to the directions of maximum variation in the row space of X. These are also called the left singular vectors, and they represent how each sample in the data is composed of different patterns. Each left singular vector corresponds to a principal component of the data.
Σ: The diagonal entries of Σ are the singular values of X. The singular values represent the amount of variation in the data that is captured by each corresponding left/right singular vector. The singular values are sorted in decreasing order, with the first singular value capturing the most variation, the second capturing the second most, and so on.
V: The columns of V are the eigenvectors of X^T X (or 1/n X^T X if we use normalized data), which correspond to the directions of maximum variation in the column space of X. These are also called the right singular vectors, and they represent the patterns in the data that contribute most to the variation. Each right singular vector corresponds to a feature of the data.
The SVD decomposition allows us to transform the input data matrix X into a lower-dimensional space, where we can retain the most important information of the data. By discarding the singular values with the smallest magnitudes, we can reduce the dimensionality of the data while retaining most of its important information. This is useful in many applications, such as image compression, feature extraction, and data visualization.

### Part 4
For part 4, we plot three V-modes onto a 3D plot. I chose to plot columns 0, 1, and 2.

![download](https://user-images.githubusercontent.com/129991497/234189078-8f9c2c7f-68cf-49f5-b366-4edbc02d7db6.png "Fig 4. 3 V-modes plotted in 3D against the 10 digits of mnist")

### 2 Digit LDA
For the 2-digit LDA, I chose to look at "4" and "5".
The accuracies were as follows:
```
Accuracy for digit 4: 0.96
Accuracy for digit 5: 0.97
```

### 3 Digit LDA
For the 3-digit LDA, I chose to look at "0", "1", and "2".

The accuracies were as follows:
```
Accuracy for digit 0: 0.91
Accuracy for digit 1: 0.95
Accuracy for digit 2: 0.96
```

### Most Difficult to Separate
In this part, I determined which two digits we the most difficult to separate using LDA. To do this I compare the accuracies of the classifier on each pair of digits.
At 93% accurate, the pair of "2" and "6" was the most difficult to separate.

### Compare Performance
In the final part of the project, I compared the accuracy of the LDA classifier against an SVM classifier and a Decision Tree classifier.

The results were as follows:
```
LDA accuracy: .97
SVM accuracy: 0.98
Decision Tree classifier accuracy: 0.96
```

The LDA and SVM classifiers both performed quite well with the SVM slightly out-performing the LDA. The Decision Tree classifier fell behind slightly.

## Summary and Conclusions
Correlation matrices help to identify relationships between variables, while eigenvectors and eigenvalues provide insights into the underlying structure of a dataset. Singular Value Decomposition (SVD) is a powerful tool for dimensionality reduction, feature extraction, and data compression.

Linear Discriminant Analysis (LDA) is another important statistical technique that is used to classify data by maximizing the separability of different classes. By finding the optimal projection of the data onto a lower-dimensional space, LDA can improve classification accuracy and help identify important features in the dataset.

Overall, the combination of these techniques provides a powerful toolkit for analyzing complex datasets and extracting meaningful insights. By using matrix operations and visualization techniques, researchers can gain a better understanding of the relationships between different variables and the underlying structure of the data.

