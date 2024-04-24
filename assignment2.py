# Python 3 program to find 
# factorial of given number
def factorial(n):
    
    # single line to find factorial
    return 1 if (n==1 or n==0) else n * factorial(n - 1) 

# Driver Code
num = 5
print("Factorial of",num,"is",factorial(num))




















# def is_palindrome(s):
#     code = s
#     if (code==code[::-1]):
#         print("TRUE")
#         return True
#     else:
#         print("FALSE")
#     return False
# print(is_palindrome('tundeednut'))





# l = [2,6,9,7,41,4]

# def getmax(l):
#     assume = l[0]
#     for i in range(1,len(l)):
#         if l[i] > assume:
#             assume = l[i]
#     return assume

# print(getmax(l))

# def f(words):
#     return [item  for item in words.split() if len(item) > 5]

# s = "Hello, my name is Daniel"
# print(f(s))

