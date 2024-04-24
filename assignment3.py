#1. A string is a collection of alpha numerical words or character in a data set
# my_list= []
# print(my_list)
str= ("daniel is a boy")
# print("boy" in str)
# print(str[2])
# print(len(str))
# str2= ("and a girl")
# new_str=str+ (" ")+ str2
# print(new_str)
# str.upper()
# print(str.upper())
# str.lower()
# print(str.lower())
# we modify strings using the string functin()
#using the if and else statement
# for str in str:
#     if "daniel" in str:
#         print("true")
#     else:
#         print("false")
# str= (str for str in str if "daniel" in str)
# print(str)
# str.split(" ")
# print(str.split(" "))
# str.strip(str)
# print(str.strip(str))
# str2= str.replace("boy","girl")
# print(str2)

# str[1:12]
# print(str[0:7])


#using loop
# for i in range(len(str)):
#     print(str[3])
# my_name= str(reversed(" "))
# print(my_name)
# function to check string is
# palindrome or not
# 
# function to check string is
# palindrome or not
def isPalindrome(s):
 
#     # Using predefined function to
#     # reverse to string print(s)
     rev = ''.join(reversed(s))
 
#     # Checking if both string are
#     # equal or not
     if (s == rev):
         return True
     return False
 
# # main function
# s = "malayalam"
# ans = isPalindrome(s)
 
# if (ans):
#     print("Yes")
# else:
#     print("No")

# str2= [ "is ", "a ", 'girl']
# my_name= str.join(str2)
# print(my_name)
# num1= 5
# num2=str(5)
# print(type(num2))

# str.isdigit()
# print(str.isdigit())

# str= ("daniel is a boy")
# new_str= str.remove(" boy")
# print(new_str)


#questions on List 
#A list is an ordered number of element
# my_list= [ ]
# print(my_list)
# my_list= ['daniel']
# print(my_list)
# my_list= ['apple','banana','mango','pawpaw']
# my_list[3]
# print(my_list[3])

#yes it can be modified using; extend,replace,remove etc
# my_list.append('watermelon')
# print(my_list)
# my_list.insert(2, 'watermelon')
# print(my_list)
#print(len(my_list))
# for item in my_list:
#     if 'mango' in my_list:
#         print("true")
#     else:
#         print("false")
# item=(item for item  in item  if "banana" in item)
# print(item)
# my_list.count('mango')
# print(my_list.count('mango'))
#my_list.pop(1)
#print(my_list.pop(1))
# my_list.remove("mango")
# print(my_list)
# mylist=['pawpaw', 'mango', 'apple']
# mylist.sort()
# print(mylist)
#my_list.reverse()
#print(my_list)
# my_list[3:20]
# print(my_list[2:20])
# for item in my_list:
#     print(my_list[0:2])
# color= ['black','blue', 'orange ']
# # num.max()
# # num.min()
# new_list=my_list + color 
# print(new_list)
#newlist= tuple(my_list)
#print(newlist())
# a=set(my_list)
# b=set(color)
# if a==b:
#     print ("my_list and color are equal ")
# else:
#     print("my_list and color are not equal " )