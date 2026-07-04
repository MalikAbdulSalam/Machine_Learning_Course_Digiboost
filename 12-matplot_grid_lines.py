import matplotlib.pyplot as plt


value1 = (1,2,3,4,5,6)


value2 = (1,2,3,4,5,60)

plt.plot(value1, value2, "o")

# add label to x 
plt.xlabel("values of value1")


#add label to y
plt.ylabel("values of value2")

# add title to plot
plt.title("Graph", loc = 'right')


# add grid to plot
# plt.grid()
#
# plt.grid(axis = 'x')   # show only x axis grid

# plt.grid(axis = 'y')	# show only y axis grid

plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

plt.show()




















#print("befor show")

#plt.show()

#print("after show")