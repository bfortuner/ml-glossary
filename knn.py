from collections import Counter
from math import sqrt

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance +=(point1[i] - point2[i]) ** 2
    return sqrt(distance)

def mean(labels):
    return sum(labels) / len(labels)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def KNN(training_data, target, k, func):
    neighbors= []
    # For each example in the training_data
    for index, data in enumerate(training_data):
        # distance between the target data and the current example from the data.
        distance = euclidean_distance(data[:-1], target)
        neighbors.append((distance, index))

    sorted_neighbors = sorted(neighbors)

    #Pick the first K entries from the sorted list
    k_nearest = sorted_neighbors[:k]

    # Get the labels of the selected K entries
    k_nearest_labels = [training_data[i][1] for distance, i in k_nearest]

    # If regression return the mean & if classification return the mode of the K labels.
    return k_nearest , func(k_nearest_labels)

def main():
    """
    # Regression Data(Column 0 : Height(inch), Column 1: Weight(lb))
    """
    reg_data = [
       [73.84, 241.89],
       [68.78, 162.31],
       [74.11, 212.74],
       [71.73, 220.04],
       [69.88, 206.34],
       [67.25, 152.21],
       [63.45, 156.39]
    ]

    target_data = [70]
    reg_k_nearest_neighbors, reg_prediction = KNN(
        reg_data, target_data, k=3, func=mean
    )
    print(reg_prediction)
    '''
    # Classification Data( Column 0: age, Column 1:like paragliding  or not )
    '''
    clf_data = [
       [26, 1],
       [20, 1],
       [22, 1],
       [19, 1],
       [28, 0],
       [33, 0],
       [30, 0],
       [50, 0],
    ]
    target_data2 = [32]
    clf_k_nearest_neighbors, clf_prediction = KNN(
        clf_data, target_data2, k=3, func=mode
    )
    print(clf_prediction)

if __name__ == '__main__':
    main()
