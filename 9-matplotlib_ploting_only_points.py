#matplotlib
#why ploting (outlier detection)


import matplotlib

# print version of matplotlib
print(matplotlib.__version__)

print("################################# ploting ##########################")



import matplotlib.pyplot as plt


x = (1,2,3,4,5,6,7,8,9, 50)


y = (1,2,3,4,5,6,7,8,9, 5)

plt.plot(x, y, 'o') # if you want to add * use it in quotes

plt.plot(x ,y,"o:",  ms = 20)

plt.show()


