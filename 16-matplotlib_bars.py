import matplotlib.pyplot as plt


value1 = (1,2,3,4,5,6,7,8,9,10)


value2 = (1,2,3,4,5,60,70, 80, 90, 100)



# plt.bar(value1,value2, color = "color_name", width = size of width of bars)

plt.bar(value1,value2, color = "#00FFFF")  # plt.barh() is used for horizontal plot and plt.bar() vertical plot
plt.show()


note:- height = tell height of bars bwtween 0 and 1 and it is only used in horizontal bars
