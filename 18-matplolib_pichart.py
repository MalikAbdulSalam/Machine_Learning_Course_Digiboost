import matplotlib.pyplot as plt

x = (10, 101, 100 , 500)
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
myexplode = [0, 0.3, 0, 0]
mycolors = ["black", "hotpink", "b", "#4CAF50"]



plt.pie(x, labels = mylabels, startangle = 90 ,explode = myexplode ,shadow = True, colors = mycolors)

plt.legend()

plt.show()