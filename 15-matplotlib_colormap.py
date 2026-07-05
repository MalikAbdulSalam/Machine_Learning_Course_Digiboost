import matplotlib.pyplot as plt


value1 = (1,2,3,4,5,6,7,8,9,10)


value2 = (1,2,3,4,5,60,70, 80, 90, 100)

colors = (0, 10, 20, 30, 40, 45 ,50,55,60,65)
size = (0, 10, 20, 30, 40, 45 ,50,55,60,65)

plt.scatter(value1, value2, c=colors, cmap='YlOrRd', s = 8 , alpha=1.0) # you may add hardcoded or use data as size : s= size

plt.colorbar()

plt.show()



