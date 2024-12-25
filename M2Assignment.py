# Your name: Jared Daniel
# Your PSU Email: jjd6385

# Assignment name: Review Math and use the related Python Packages
# Module number: 2


from typing import List

##############   From Chpater 4. Linear Algebra    #############  

import numpy as np  

#1.

#def add(v: Vector, w: Vector) -> Vector:
#    """Adds corresponding elements"""
#    assert len(v) == len(w), "vectors must be the same length"
#
#    return [v_i + w_i for v_i, w_i in zip(v, w)]
#
#assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

###### Your codes to replace above cell of codes: 

def add(v, w):
    """Adds corresponding elements"""
    return np.add(v, w)
print( add([1, 2, 3], [4, 5, 6]))

###########################

#2
#def subtract(v: Vector, w: Vector) -> Vector:
#    """Subtracts corresponding elements"""
#    assert len(v) == len(w), "vectors must be the same length"
#
#    return [v_i - w_i for v_i, w_i in zip(v, w)]
#
#assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

###### Your codes to replace above cell of codes:  

def subtract(v, w):
    """Subtracts corresponding elements"""
    return np.subtract(v, w)
print( subtract([5, 7, 9], [4, 5, 6]))

###########################

#3

#def vector_sum(vectors: List[Vector]) -> Vector:
#    """Sums all corresponding elements"""
#    # Check that vectors is not empty
#    assert vectors, "no vectors provided!"
#
#    # Check the vectors are all the same size
#    num_elements = len(vectors[0])
#    assert all(len(v) == num_elements for v in vectors), "different sizes!"
#
#    # the i-th element of the result is the sum of every vector[i]
#    return [sum(vector[i] for vector in vectors)
#            for i in range(num_elements)]
#
#assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

###### Your codes to replace above cell of codes:  

def vector_sum(vectors):
    """Sums all corresponding elements"""
    return np.sum(vectors, axis=0)
print( vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]))

###########################

#4

#def scalar_multiply(c: float, v: Vector) -> Vector:
#    """Multiplies every element by c"""
#    return [c * v_i for v_i in v]
#
#assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

###### Your codes to replace above cell of codes: 

def scalar_multiply(c, v):
    """Multiplies every element by c"""
    v = np.array(v)
    return c * v
print( scalar_multiply(2, [1, 2, 3]))

###########################

#5

#def vector_mean(vectors: List[Vector]) -> Vector:
#    """Computes the element-wise average"""
#    n = len(vectors)
#    return scalar_multiply(1/n, vector_sum(vectors))
#
#assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


###### Your codes to replace above cell of codes:  

def vector_mean(vectors):
    """Computes the element-wise average"""
    vectors = np.array(vectors)
    return np.mean(vectors, axis=0, dtype=np.int32)
print( vector_mean([[1, 2], [3, 4], [5, 6]]))

###########################

#6

#def dot(v: Vector, w: Vector) -> float:
#    """Computes v_1 * w_1 + ... + v_n * w_n"""
#    assert len(v) == len(w), "vectors must be same length"
#
#    return sum(v_i * w_i for v_i, w_i in zip(v, w))
#
#assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6
#

###### Your codes to replace above cell of codes:  

def dot(v, w) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    v = np.array(v)
    w = np.array(w)
    return np.dot(v, w)
print( dot([1, 2, 3], [4, 5, 6]))

###########################

#7

#def sum_of_squares(v: Vector) -> float:
#    """Returns v_1 * v_1 + ... + v_n * v_n"""
#    return dot(v, v)
#
#assert sum_of_squares([1, 2, 3]) == 14  # 1 * 1 + 2 * 2 + 3 * 3
#

###### Your codes to replace above cell of codes: 

def sum_of_squares(vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    vector = np.array(vector)
    return np.sum(vector ** 2)
print( sum_of_squares([1, 2, 3]))

###########################

#8

#def magnitude(v: Vector) -> float:
#    """Returns the magnitude (or length) of v"""
#    return math.sqrt(sum_of_squares(v))   # math.sqrt is square root function
#
#assert magnitude([3, 4]) == 5

###### Your codes to replace above cell of codes: 

def magnitude(vector) -> float:
    """Returns the magnitude (or length) of v"""
    vector = np.array(vector)
    return np.linalg.norm(vector)
print(magnitude([3, 4]))

###########################

#9

#def squared_distance(v: Vector, w: Vector) -> float:
#    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
#    return sum_of_squares(subtract(v, w))

###### Your codes to replace above cell of codes: 

def squared_distance(v, w):
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    v = np.array(v)
    w = np.array(w)
    return np.sum((v - w) ** 2)
print(squared_distance([1, 2, 3], [4, 5, 6]))

###########################

#10

#def distance(v: Vector, w: Vector) -> float:
#    """Computes the distance between v and w"""
#    return math.sqrt(squared_distance(v, w))
#
#
###### Your codes to replace above cell of codes: 

def distance(v,w):
    """Computes the distance between v and w"""
    v = np.array(v)
    w = np.array(w)
    return np.linalg.norm(v - w)
print(distance([1, 2, 3], [4, 5, 6]))

###########################


#11

#def shape(A: Matrix) -> Tuple[int, int]:
#    """Returns (# of rows of A, # of columns of A)"""
#    num_rows = len(A)
#    num_cols = len(A[0]) if A else 0   # number of elements in first row
#    return num_rows, num_cols
#
#assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2 rows, 3 columns
    
###### Your codes to replace above cell of codes: 
    
def shape(A):
    """Returns (# of rows of A, # of columns of A)"""
    A = np.array(A)
    return A.shape
print(shape([[1, 2, 3], [4, 5, 6]]))

###########################


#############  From Chpater 5. Statistics  #############
import pandas as pd

num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

#from collections import Counter
#import matplotlib.pyplot as plt
#
#friend_counts = Counter(num_friends)
#xs = range(101)                         # largest value is 100
#ys = [friend_counts[x] for x in xs]     # height is just # of friends
#plt.bar(xs, ys)
#plt.axis([0, 101, 0, 25])
#plt.title("Histogram of Friend Counts")
#plt.xlabel("# of friends")
#plt.ylabel("# of people")
#plt.show()
#
#num_points = len(num_friends)               # 204
#
#
#assert num_points == 204
#
#largest_value = max(num_friends)            # 100
#smallest_value = min(num_friends)           # 1
#
#
#assert largest_value == 100
#assert smallest_value == 1
#
#sorted_values = sorted(num_friends)
#smallest_value = sorted_values[0]           # 1
#second_smallest_value = sorted_values[1]    # 1
#second_largest_value = sorted_values[-2]    # 49
#
#
#assert smallest_value == 1
#assert second_smallest_value == 1
#assert second_largest_value == 49



#12 

# Hint: convert list to pandas object Series

#def mean(xs: List[float]) -> float:
#    return sum(xs) / len(xs)
#
#mean(num_friends)   # 7.333333
#
#
#assert 7.3333 < mean(num_friends) < 7.3334

###### Your codes to replace above cell of codes: 

def mean(xs: List[float]) -> float:
    series = pd.Series(xs)  # creating series 
    return series.mean()

print(mean(num_friends))  # 7.333333

###########################

#13

# The underscores indicate that these are "private" functions, as they're
# intended to be called by our median function but not by other people
# using our statistics library.
#def _median_odd(xs: List[float]) -> float:
#    """If len(xs) is odd, the median is the middle element"""
#    return sorted(xs)[len(xs) // 2]
#
#def _median_even(xs: List[float]) -> float:
#    """If len(xs) is even, it's the average of the middle two elements"""
#    sorted_xs = sorted(xs)
#    hi_midpoint = len(xs) // 2  # e.g. length 4 => hi_midpoint 2
#    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2
#
#def median(v: List[float]) -> float:
#    """Finds the 'middle-most' value of v"""
#    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)
#
#assert median([1, 10, 2, 9, 5]) == 5
#assert median([1, 9, 2, 10]) == (2 + 9) / 2
#
#
#assert median(num_friends) == 6

#def median(v: List[float]) -> float:
#    """Finds the 'middle-most' value of v"""
#    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)


###### Your codes to replace above cell of codes: 

def median(xs: List[float]) -> float:
    series = pd.Series(xs)  # creating series 
    return series.median()

print(median([1, 10, 2, 9, 5]))
print(median([1, 9, 2, 10]))
print(median(num_friends))

###########################


#14

#def quantile(xs: List[float], p: float) -> float:
#    """Returns the pth-percentile value in x"""
#    p_index = int(p * len(xs))
#    return sorted(xs)[p_index]
#
#assert quantile(num_friends, 0.10) == 1
#assert quantile(num_friends, 0.25) == 3
#assert quantile(num_friends, 0.75) == 9
#assert quantile(num_friends, 0.90) == 13
#
###### Your codes to replace above cell of codes: 

def quantile(xs: List[float], p: float) -> float:
    series = pd.Series(xs)  # creating series 
    return series.quantile(p)
print(quantile(num_friends, 0.10))
print(quantile(num_friends, 0.25))
print(quantile(num_friends, 0.75))
print(quantile(num_friends, 0.90))

###########################

#15
#Hint You may use Pandas DataFrame: mode() function

#def mode(x: List[float]) -> List[float]:
#    """Returns a list, since there might be more than one mode"""
#    counts = Counter(x)
#    max_count = max(counts.values())
#    return [x_i for x_i, count in counts.items()
#            if count == max_count]
#
#assert set(mode(num_friends)) == {1, 6}
#
###### Your codes to replace above cell of codes: 

def mode(x: List[float]) -> List[float]:
    """Returns a list, since there might be more than one mode"""
    series = pd.Series(x)  # creating series 
    return series.mode()
print(mode(num_friends))

###########################

# No related fuction in numpy or pandas, you can skip

## "range" already means something in Python, so we'll use a different name
#def data_range(xs: List[float]) -> float:
#    return max(xs) - min(xs)
#
#assert data_range(num_friends) == 99

def data_range(xs: List[float]) -> float:
    """Compute the range of a list"""
    series = pd.Series(xs)  
    return series.max() - series.min()
print(data_range(num_friends))

#16 

# You can define variance directy by using pandas

#from scratch.linear_algebra import sum_of_squares
#
#def de_mean(xs: List[float]) -> List[float]:
#    """Translate xs by subtracting its mean (so the result has mean 0)"""
#    x_bar = mean(xs)
#    return [x - x_bar for x in xs]
#
#def variance(xs: List[float]) -> float:
#    """Almost the average squared deviation from the mean"""
#    assert len(xs) >= 2, "variance requires at least two elements"
#
#    n = len(xs)
#    deviations = de_mean(xs)
#    return sum_of_squares(deviations) / (n - 1)
#
#assert 81.54 < variance(num_friends) < 81.55

###### Your codes to replace above cell of codes: 

def variance(xs: List[float]) -> float:
    """Almost the average squared deviation from the mean"""
    series = pd.Series(xs)  # creating series 
    return series.var(ddof=1)
print( variance(num_friends))

###########################

#17

#import math
#
#def standard_deviation(xs: List[float]) -> float:
#    """The standard deviation is the square root of the variance"""
#    return math.sqrt(variance(xs))
#
#
###### Your codes to replace above cell of codes: 

def standard_deviation(xs: List[float]) -> float:
    """The standard deviation is the square root of the variance"""
    series = pd.Series(xs)  # creating series 
    return series.std(ddof=1)
print( standard_deviation(num_friends))

###########################

# Skip this
#def interquartile_range(xs: List[float]) -> float:
#    """Returns the difference between the 75%-ile and the 25%-ile"""
#    return quantile(xs, 0.75) - quantile(xs, 0.25)
#
#assert interquartile_range(num_friends) == 6
#
#


daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

daily_hours = [dm / 60 for dm in daily_minutes]

#18

#from scratch.linear_algebra import dot
#
#def covariance(xs: List[float], ys: List[float]) -> float:
#    assert len(xs) == len(ys), "xs and ys must have same number of elements"
#
#    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)
#
#assert 22.42 < covariance(num_friends, daily_minutes) < 22.43
#assert 22.42 / 60 < covariance(num_friends, daily_hours) < 22.43 / 60

###### Your codes to replace above cell of codes: 

def covariance(xs: List[float], ys: List[float]) -> float:
    """ Measuring the covariance between 2 lists"""
    xs = np.array(xs)
    ys = np.array(ys)
    return np.cov(xs, ys)[0][1]
print(covariance(num_friends, daily_minutes))

###########################

#19

#def correlation(xs: List[float], ys: List[float]) -> float:
#    """Measures how much xs and ys vary in tandem about their means"""
#    stdev_x = standard_deviation(xs)
#    stdev_y = standard_deviation(ys)
#    if stdev_x > 0 and stdev_y > 0:
#        return covariance(xs, ys) / stdev_x / stdev_y
#    else:
#        return 0    # if no variation, correlation is zero
#
#assert 0.24 < correlation(num_friends, daily_minutes) < 0.25
#assert 0.24 < correlation(num_friends, daily_hours) < 0.25

###### Your codes to replace above cell of codes: 

def correlation(xs: List[float], ys: List[float]) -> float:
    """Calculating the correlation between 2 lists"""
    xs = np.array(xs)
    ys = np.array(ys)
    return np.corrcoef(xs, ys)[0][1]
print(correlation(num_friends, daily_minutes))

###########################

###############

outlier = num_friends.index(100)    # index of outlier

num_friends_good = [x
                    for i, x in enumerate(num_friends)
                    if i != outlier]

daily_minutes_good = [x
                      for i, x in enumerate(daily_minutes)
                      if i != outlier]

daily_hours_good = [dm / 60 for dm in daily_minutes_good]

assert 0.57 < correlation(num_friends_good, daily_minutes_good) < 0.58
assert 0.57 < correlation(num_friends_good, daily_hours_good) < 0.58

#############  From Chpater 6. Probability  #############

import scipy.stats as ss
import matplotlib.pyplot as plt

# 20
#def uniform_cdf(x: float) -> float:
#    """Returns the probability that a uniform random variable is <= x"""
#    if x < 0:   return 0    # uniform random is never less than 0
#    elif x < 1: return x    # e.g. P(X <= 0.4) = 0.4
#    else:       return 1    # uniform random is always less than 1
#
#SQRT_TWO_PI = math.sqrt(2 * math.pi)

###### Your codes to replace above cell of codes: 

def uniform_cdf(x:float) -> float:
    ss.uniform.cdf(x)

###########################

# cdf(x, loc=0, scale=1)

# 21
#import math
#def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
#    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

###### Your codes to replace above cell of codes: 

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return ss.norm.pdf(x, loc=mu, scale=sigma)

###########################

### th following codes for drawing the graphs

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()


# 22
#def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
#    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

###### Your codes to replace above cell of codes: 

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return ss.norm.cdf(x, loc=mu, scale=sigma)

###########################


### th following codes for drawing the graphs
xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4) # bottom right
plt.title("Various Normal cdfs")
plt.show()

