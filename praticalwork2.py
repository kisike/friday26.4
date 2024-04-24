# def greet_user(s):
#     s = ('Hello welcome to digital fortress')
#     return s
# name= ("Daniel")
# print(greet_user(name))

# def find_max(numbers):
#     max= numbers[0]
#     for numbers in numbers:
#         if max > numbers:
#             max= numbers
#     return max 
    
# def find_max(x):
#     max= x[0]
#     for x in x:
#         if x > max:
#             max = x
#     return max  
# from utils import find_max
# import utils 
# numbers = (2,3,4,5,6,7,8,9)
# new = utils.find_max(numbers)
# print(new)

# import converter
# print(converter.kg_to_lbs)

# import random
# for i in range(3):
#     print(random.randint(10, 20))

# import random
# member= ['mary', 'john', 'daniel']
# leader= random.choice(member)
# print(leader)

# import random

# class Dice:
#     def roll(self):
#        first= random.randint(1, 6)
#        second= random.randint(1, 6)
#        return first, second

# dice= Dice()
# print(dice.roll())


# tunde = open("tunde.txt", "r")
# # print(tunde.read())
# # tunde.close()
# print(tunde)
# tunde = open("tunde2.txt", "a")
# tunde.write("We are in python class")
# tunde.close()
# tunde2= open("tunde3.txt", "w")
# tunde2.write("python is a flesxible language")
# tunde2.close()
# tunde2= open("tunde4.txt", "w")
# tunde2.write("come and learn from digital fortress")
# tunde2.close()
# tunde = open(r"C:\Users\PC\Sule daniel\praticalwork2\tunde2>", "r")
# print(tunde.read())
# tunde.close

import os 
os.remove("tunde.txt")

