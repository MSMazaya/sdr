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

class PolynomailRegression() :
     
    def __init__( self, degree, learning_rate, iterations ) :
         
        self.degree = degree
         
        self.learning_rate = learning_rate
         
        self.iterations = iterations
         
    # function to transform X
     
    def transform( self, X ) :
         
        # initialize X_transform
         
        X_transform = np.ones( ( self.m, 1 ) )
         
        j = 0
     
        for j in range( self.degree + 1 ) :
             
            if j != 0 :
                 
                x_pow = np.power( X, j )
                 
                # append x_pow to X_transform
                 
                X_transform = np.append( X_transform, x_pow.reshape( -1, 1 ), axis = 1 )
 
        return X_transform  
     
    # function to normalize X_transform
     
    def normalize( self, X ) :
         
        X[:, 1:] = ( X[:, 1:] - np.mean( X[:, 1:], axis = 0 ) ) / np.std( X[:, 1:], axis = 0 )
         
        return X
         
    # model training
     
    def fit( self, X, Y ) :
         
        self.X = X
     
        self.Y = Y
     
        self.m, self.n = self.X.shape
     
        # weight initialization
     
        self.W = np.zeros( self.degree + 1 )
         
        # transform X for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
         
        X_transform = self.transform( self.X )
         
        # normalize X_transform
         
        X_normalize = self.normalize( X_transform )
                 
        # gradient descent learning
     
        for i in range( self.iterations ) :
             
            h = self.predict( self.X )
         
            error = h - self.Y
             
            # update weights
         
            self.W = self.W - self.learning_rate * ( 1 / self.m ) * np.dot( X_normalize.T, error )
         
        return self
     
    # predict
     
    def predict( self, X ) :
      
        # transform X for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
         
        X_transform = self.transform( X )
         
        X_normalize = self.normalize( X_transform )
         
        return np.dot( X_transform, self.W )


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
        model = PolynomailRegression( degree = i, learning_rate = 0.01, iterations = 500 )
 
        model.fit( X_train, Y_train )
    
        y_pred = model.predict( X_train )
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
