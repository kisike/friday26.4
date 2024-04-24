# num1=int(input("Enter your number: "))
# # num2= 2
# if (num1 % 2)==0:
#         print("the number is even")
# else :
#         print("the number provided is odd")


# user= int(input("Enter your age: \n"))
# Age= 20
# if user ==Age or user > Age :
#         print(" You are eligible to vote")
# else:
#         print("You are not eligible to vote ")

# user= int(input("Enter your number: \n"))
# if (user > 0):
#     print("positive")
# elif (user < 0):
#     print("negative")
# else:
#     user=0
#     print("Zero")

# user= int(input("Enter your number: \n"))
# if (user % 3):
#     print("Divisible by 3') 
# elif (user / 5):
#     print('Divisible by 5')
# else:
#     print("false")



# user1= int(input("Enter your first number: \n"))
# user2= int(input("Enter your second number: \n"))
# if (user1 > user2):
#     print(f'{user1} is the largest)
# else:
#     print(f'{user2} is the largest) 


# grade= ('A', 'B', 'C', 'D', 'E','F')
# Grades= input("Enter your grade: \n")
# for i in grade:   
#  if Grades== ('A' or'B'or'C'):
#         print("PASS")
# else: 
#         print("FAIL")



# user_salary= float(input("enter your salary: \n")) 
# if user_salary <= 5000:
#     tax_percentage= 10
# elif user_salary == 5001 or range (0,10000):
#     tax_percentage= 20
# elif user_salary > 10000:
#     tax_percentage = 30

# tax_amount= (user_salary * tax_percentage) / 100
# net_salary= user_salary - tax_amount
# print(net_salary)

# salary= int(input("Enter your salary    \n"))
# tax1= (salary * 10)/ 100
# tax2= (salary * 20)/ 100
# tax3= (salary * 30)/ 100
# if salary <= 5000:
#     print(f'{tax1}')
# elif salary == 5001 or range (0,10000):
#     print(f'{tax2}')
# else:
#     print(f'{tax3}')









# user1=int(input("Enter your first number: \n"))
# user2=int(input("Enter your second number: \n"))
# sum= user1 + user2
# if (sum > 100):
#     print("Sum is greater than 100")
# else:
#     print("sum is not greater than 100")

# x=int(input("enter your first number:  \n"))
# y=int(input("Enter your second number:  \n"))
# z= int(input("enter your third number: \n"))
# if (x > y) and (x > z):
#     print(x)
# elif (y > x) and (y > z):
#     print(y)
# elif (z > x) and (z > y):
#     print(z)
# else:
#     print("none")

# user_age=int(input("Enter your age:  \n"))
# required_age= 60 
# if user_age < required_age:
#     print("you are not eligible for a senior citizen discount")
# elif user_age == required_age:
#      print("you are eligible for senior a citizen discount")
# else:
#     print("you are not eligible for senior a citizen discount")



# 90 to 100= excellent
# 80 to 89= graade b 
# 70 to 79= grade c 
# 60 - 69 = grade d
# 50 to 59 = fail
# 0 to 49 = unkown result
user = int(input("Enter your number:  \n"))
if user in range (90, 100):
    print("Excellent")
elif user in range (80, 89):
    print("Grade B")
elif user in range (70, 79):
    print("grade c")
elif user in range (60, 69):
    print("grade d")
elif user in range (50, 59):
    print("fail")
elif user in range (0, 49):
    print("unkown result") 





















