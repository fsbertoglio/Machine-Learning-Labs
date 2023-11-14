
TRAIN_2F_NORM = "../Dados_Normalizados_2Features/TrainingData_2F_Norm.txt"
TEST_2F_NORM = "../Dados_Normalizados_2Features/TestingData_2F_Norm.txt"

TRAIN_11F_NORM = "../Dados_Normalizados_11Features/TrainingData_11F_Norm.txt"
TEST_11F_NORM = "../Dados_Normalizados_11Features/TestingData_11F_Norm.txt"

TRAIN_2F_ORI = "../Dados_Originais_2Features/TrainingData_2F_Original.txt"
TEST_2F_ORI = "../Dados_Originais_2Features/TestingData_2F_Original.txt"

TRAIN_11F_ORI = "../Dados_Originais_11Features/TrainingData_11F_Original.txt"
TEST_11F_ORI = "../Dados_Originais_11Features/TestingData_11F_Original.txt"


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the data from file
training_data = np.genfromtxt(TRAIN_2F_ORI, delimiter='\t', skip_header=1, usecols=(0,1,2,3))
test_data = np.genfromtxt(TEST_2F_ORI, delimiter='\t', skip_header=1, usecols=(0,1,2,3))

# Split the data into X and Y
X_train = training_data[:, 1:-1]
Y_train = training_data[:, -1]

X_test = test_data[:, 1:-1]
Y_test = test_data[:, -1]


print("DADOS ORIGINAIS: 2 FEATURES")
for k in range(1, 8, 2):    
    # Create the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the data
    knn.fit(X_train, Y_train)

    # print(f" Output : {knn.predict(X_test)}")
    # print(f" Expected : {Y_test}")
    print(f" Accuracy for k = {k}: {knn.score(X_test, Y_test)}")

    # for i in X_test:
    #     kneighbors = knn.kneighbors([i], k, True)
    #     print(kneighbors)



# Load the data from file
training_data = np.genfromtxt(TRAIN_2F_NORM, delimiter='\t', skip_header=1, usecols=(0,1,2,3))
test_data = np.genfromtxt(TEST_2F_NORM, delimiter='\t', skip_header=1, usecols=(0,1,2,3))

# Split the data into X and Y
X_train = training_data[:, 1:-1]
Y_train = training_data[:, -1]

X_test = test_data[:, 1:-1]
Y_test = test_data[:, -1]    

print("DADOS NORMALIZADOS: 2 FEATURES")
for k in range(1, 8, 2):    
    # Create the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the data
    knn.fit(X_train, Y_train)

    # print(f" Output : {knn.predict(X_test)}")
    # print(f" Expected : {Y_test}")
    print(f" Accuracy for k = {k}: {knn.score(X_test, Y_test)}")

    # for i in X_test:
    #     kneighbors = knn.kneighbors([i], k, True)
    #     print(kneighbors)
