
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib.colors import LinearSegmentedColormap

def importIris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def plotIris(data, target):
    # Taken with minor changes from http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
    # first two dimensions of X
    X = data[:,:2]
    y = target
    # Plot the training points using a colormap
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
                edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xticks(())
    plt.yticks(())
    plt.show()

    # Second two dimensions of X
    X = data[:,2:]
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
                edgecolor='k')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.xticks(())
    plt.yticks(())
    plt.show()  

def oneVsAll(y):
    # turns a multi-categorical y vector into a matrix of one-vs-all single category y vectors
    ylist = []
    for yval in np.unique(y):
        ylist.append( (y==yval).astype(int) )
    return np.vstack(ylist).T

def sigmoid(x):
    # returns the sigmoid of an input x
    return np.exp(x) / ( 1 + np.exp(x))

def softmaxClassifier(z):
    # for each sample, returns the max and location of the max 
    #     of the softmax of the predictions from each classifier
    confs = np.zeros((z.shape[0]))
    ys = np.zeros((z.shape[0]))
    for s in range(z.shape[0]):
        zs = z[s,:]
        softmx = np.exp(zs)/np.sum(np.exp(zs))
        confs[s] = np.max(softmx)
        ys[s] = np.where(softmx == confs[s])[0][0]
    return ys

def initialize(x, y):
    # standardizes the x data, add x0 bias feature to x
    # applies one-vs-all to y
    # generates the initial random weights
    x = (x - x.mean())/x.std()
    x = np.concatenate((np.ones([x.shape[0],1]),x),1)
    y = oneVsAll(y)
    weights = np.random.rand(y.shape[1],x.shape[1])
    return x, y, weights
    
def predict(x, weights):
    # returns the prediction value for a given set of features (x) and weights
    z = np.dot(x, weights.T)
    return sigmoid(z)

def classify(predictions):
    #returns the category of the prediction based on a decision boundary of 0.5, i.e.:
    #   1 if the activation is greater than or equal to 0.5, else return 0
    return (predictions >= 0.5).astype(int)

def costFunction(y, predictions):
    # returns the cross entropy cost of a particular model's activations
    # to interpret the equation, remember y is only either 1 or 0
    return -1*np.mean( y*np.log(predictions) + (1-y)*np.log(1-predictions) )

def costFunctionGradient(y, predictions, x):
    # returns the derivative of the cost function w.r.t. the parameters weights
    # for each model
    return np.dot(x.T, (predictions-y)).T
    
def cgStep(x, y, weights, learn_rate):
    # takes a single batch conjugate gradient step
    predictions = predict(x, weights)
    gradient = costFunctionGradient(y, predictions, x)/x.shape[0]
    return (weights - gradient*learn_rate)

def accuracy(predicted_labels, actual_labels):
    # gives the percentage of correct labels
    diff = predicted_labels - actual_labels
    return 1.0 - (diff != 0).astype(int).mean()

def train(x, y, y_multiclass, weights, learn_rate, iterations, plot = True):
    # trains the logistic regression function using simple conjugate gradient steps of the cost function
    # multiple classification problem
    # only directly uses y_multiclass for the accuracy, however, 
    #   all other math is on the binary-model level by using matrix algebra
    cost_record = []
    acc_record = []
    for i in range(iterations):
        weights = cgStep(x,y,weights,learn_rate)
        cost_record.append(costFunction(y,predict(x,weights)))
        acc_record.append(accuracy(softmaxClassifier(predict(x,weights)),y_multiclass))
        if i % int(iterations/5) == 0:
            print("Iteration {}  Cost {:.3f}   Acc: {:.2f}".format(
                i, costFunction(y, predict(x, weights)), accuracy(softmaxClassifier(predict(x, weights)), y_multiclass) ))
    print("Final: Iteration {}  Cost {:.3f}   Acc: {:.2f}".format(
            i, costFunction(y, predict(x, weights)), accuracy(softmaxClassifier(predict(x, weights)), y_multiclass) ))
    if plot:
        plotTraining(cost_record, acc_record)
    return weights, cost_record, acc_record


def plotTraining(cost_record, acc_record):
    # use two axes to simultaneously plot the accuracy and the cost
    # first axis
    fig, ax1 = plt.subplots()
    plt.title('Training Record', fontsize = 16)
    color = 'tab:blue'
    ax1.plot(range(len(cost_record)),cost_record,'b-', lw = 3 )
    ax1.set_xlabel('Iterations', fontsize = 14)
    ax1.set_ylabel('Cost', fontsize = 14)
    ax1.tick_params(axis='y', labelcolor=color, size = 8, labelsize = 12)
    ax1.tick_params(axis='x', size = 12)
    # second axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(range(len(acc_record)),acc_record,'r-', lw = 3)
    ax2.set_ylabel('Accuracy', size = 14)
    ax2.set_ylim((0,1))
    ax2.tick_params(axis='y', labelcolor=color, size = 8, labelsize = 12)
    # bring it together
    plt.tight_layout()
    plt.show()