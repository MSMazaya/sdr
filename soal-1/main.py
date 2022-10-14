import numpy as np
import random
import matplotlib.pyplot as plt
f = lambda x:  5*x**3 - x**2 + x 

def gaussian_noise(x,mean,std):
    noise = np.random.normal(mean, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 

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

    x_train = np.array([[i] for i in x if i < 0 or i > 2])
    # x_test = np.array([[i] for i in x if i > 0 or i < 2])
    
    y_train = np.array([f(i) for i in x if i < 0 or i > 2])
    # y_test = np.array([f(i) for i in x if i > 0 or i < 2])

    model = PolynomailRegression( degree = 9, learning_rate = 0.01, iterations = 500 )
 
    model.fit( x_train, y_train )
    
    y_pred = model.predict( x_train )
#    
    plt.scatter( x_train, y_train, color = 'blue' )
   
    plt.plot( x_train, y_pred, color = 'orange' )
   
    plt.title( 'X vs Y' )
   
    plt.xlabel( 'X' )
   
    plt.ylabel( 'Y' )
   
    plt.show()

if __name__ == "__main__":
    main()
