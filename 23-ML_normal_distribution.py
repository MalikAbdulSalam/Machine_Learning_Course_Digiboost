import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)

x = (1,2,3,4,5,6,7,8,9)
print(x)
plt.hist(x, 100)
plt.show()