import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

f = lambda x:  5*x**3 - x**2 + x 

def gaussian_noise(mean,std):
    noise = np.random.normal(mean, std)
    return noise 

def generate_random(start, end, n):
    return [random.randint(start, end) for _ in range(n)]

     
def normalize(x) :
    x[:, 1:] = ( x[:, 1:] - np.mean( x[:, 1:], axis = 0 ) ) / np.std( x[:, 1:], axis = 0 )
    return x

def polinomialFitting(X, Y, degree, learning_rate, iterations ) :
    m, _ = X.shape
    W = np.zeros( degree + 1 )

    X_transform = np.ones( ( m, 1 ) )
    j = 0
    for j in range(degree+1) :
        if j != 0 :
            x_pow = np.power( X, j )
            X_transform = np.append( X_transform, x_pow.reshape( -1, 1 ), axis = 1 )

    X_normalize = normalize(X_transform)
 
    for _ in range(iterations) :
        X_transform = np.ones( ( m, 1 ) )
        j = 0
        for j in range(degree+1) :
            if j != 0 :
                x_pow = np.power( X, j )
                X_transform = np.append( X_transform, x_pow.reshape( -1, 1 ), axis = 1 )

        X_normalize = normalize(X_transform)
        h = np.dot(X_transform, W)
        error = h - Y
        W = W - learning_rate * ( 1 / m ) * np.dot( X_normalize.T, error )

    def predict(X) :
        X_transform = np.ones( ( m, 1 ) )
        j = 0
        for j in range(degree+1) :
            if j != 0 :
                x_pow = np.power( X, j )
                X_transform = np.append( X_transform, x_pow.reshape( -1, 1 ), axis = 1 )
        X_normalize = normalize(X_transform)
        return np.dot(X_transform, W)

    return predict 

def main():
    random.seed(13320028)
    x = sorted(generate_random(-5, 5, 100))

    X = np.array([[i] for i in x if i < 0 or i > 2])

    mean = 0
    std = np.sqrt(300)
    Y = np.array([f(i) + gaussian_noise(mean, std) for i in x if i < 0 or i > 2])
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=13320028)

    plt.scatter(X_test, Y_test, color='red')

    min = sys.maxsize
    index = 1

    for i in range(1, 9):
        predict = polinomialFitting(X_train, Y_train, i, 0.01, 1000)
 
        y_pred = predict(X_train)
        error = np.sum((y_pred - Y_train)**2)

        if error < min:
            index = i
            min = error
        
        plt.plot(sorted(X_train), sorted(y_pred), label='degree = {}, error = {}'.format(i, error))
   

    plt.title('Polynomial Regression, with result of degree as best fit = {}'.format(index))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
