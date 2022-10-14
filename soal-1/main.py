import numpy as np
import random

f = lambda x:  5*x**3 - x**2 + x 

def gaussian_noise(x,mean,std):
    noise = np.random.normal(mean, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 

def generate_random(start, end, n):
    return [random.randint(start, end) for _ in range(n)]

def main():
    random.seed(13320028)
    x = generate_random(-5, 5, 100)

    x_train = [i for i in x if i < 0 and i > 2]

if __name__ == "__main__":
    main()
