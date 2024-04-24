# # def add(a, b):
# #   return a + b
# # print(add(8,5))

# # def str_length():
# #     my_string = input("Enter a string: ")
# #     return len(my_string)
# # print(str_length())

# def isprime(n): 
#       if n <= 1:
#        return False
#       elif n<= 3:
#          return True
#       elif n % 2 == 0 or n % 3 == 0:
#          return False
#       i = 2
#       while i * i <= n:
#          if n % i == 0 or n % ( i * 2) == 0:
#             return False
#          i += 1
#          return True 
# print(isprime(21))
 
#      if n % 2 == 0: 
 
#         return False 
 
#      if n % 3 == 0: 
 
#         return True
# print(isprime(21))

# def list_sum(l):
#   total = 0
#   for nums in l:
#     total = total + nums
#   return total

# my_list = [6,5,3,4,8]

# print (list_sum(my_list))

# def even_num(s):
#     return [i for i in s if i % 2 == 0]
# # numbers= int(input("Enter your numbers:  \n"))
# nums= (2,3,4,5,6,7)
# print(even_num(nums))

# def factorial(n):
    
#     # single line to find factorial
#     return 1 if (n==1 or n==0) else n * factorial(n - 1)


# # Driver Code
# num = 5
# print(f"{factorial(num)}")

# def is_palindrome(s):
#     string = s
#     if (string==string[::-1]):
#         return True
#     else:
#        return False
# print(is_palindrome('tundeednut'))

# def f(sentence):
#     return [i for i in sentence.split() if len(i) > 5]

# s = "Hello, my name is Stephan"
# print(f(s))


# def sentence():
#     s= ['apple', "banana", " car", "yola", "avocado", "grape", "mango"]
#     for i in s:
#         if len(i)> 5:
#             print(i)
# print(sentence())
# l = [2,6,9,7,41,4]

# def getmax(l):
#     assume = l[0]
#     for i in range(1,len(l)):
#         if l[i] > assume:
#             assume = l[i]
#     return assume

# print(getmax(l))
     


