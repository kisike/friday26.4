#unpacking turple

# color=(" blue", "yellow", "green")
# (yam,banana,orange)= color
# print(yam)
# print(color)

#color=(" blue", "yellow", "green", "purple", "red", "pink", "black")
# (yam,*banana,orange)= color
# print(orange)
# print(banana)
# for i in color:
#     print(i[2])

#x=0
#hile x <= 20:
#    print(x)

    
# for i in range(0, 21, 2):
#     print(i)
# color=(" blue", "yellow", "green", "purple", "red", "pink", "black")
# for v in range(len(color)):
#     print(color[v])


#joining two tuple 
# color= (" blue", "yellow", "green", "purple", "red", "pink", "black")
# fruit= ("banana","mango")
# fr= color+fruit
# print(fr)

#set
#myset= {20,30,25,70,20,25,35}
# print(myset)
# print(type(myset))
# color=set((" blue", "yellow", "green", "purple", "red", "pink", "black"))
# print(type(color))    
# myset= {" blue", "yellow", "green", False,0,1,True,2}   
# print(myset)  
#print("green" in color)
# color.add("cream")
# print(color)
# color.remove("purple")
# print(color)
#color.pop()
#print(color)
# color.clear()
# print(color)
# # del color
# # print(color)
# myfruits= {"mango", "pineapple", "orange"}
# # print(f'{color} {myfruits}')
# myitem= color.union(myfruits)
# print(myitem)



#dictionary
# employee= {
#     "name":"Emeka",
#     "state":"lagos state",
#     "job": "software development"
# }
# print(employee["name"])

# employee=[
# {
#     "name":"Emeka",
#     "state":"lagos state",
#     "job": "software development"
# },     {
    # "name": "kike",  
#     "state":"ogun state",
#     "job": "data analyst"

# },    {
#     "name": "tade",
#     "state":"Rivers state",
#     "job":"programmer"
# },    {
#     "name": "samson",
#     "state": "Abia state",
#     "job": "graphics designer"
# }  
# ]
# # print(employee[2]["name"])
# employee[3]["state"]="FCT"
# # print(employee[3])
# employee= {
#     "name":"Emeka",
#     "state":"lagos state",
#     "job": "software development"
# }
# print(employee["name"])
# print(employee.get("state"))
# employee.update[{"name"}:{"victor"}]


# def greeting():
#     print("welcome to digital")
# greeting()
# def myclass(x):
#     return x ** 2
# print(myclass(10))


# any number divisble by 5 should print fizz if not print buzz
# # counter= 0
# # while counter <=21:
    # if i % 2==0: 
    #   print("fizz")
    # elif i % 5==0:
    #      print("buzz")
    

# for i in range (0,21):
#     if i % 2==0: 
#         print("fizz")
#     elif i % 5==0:
#         print("buzz")
#     else:
#         print(i)
    
# myitem= ["Training", "class", "item", "tech", "throwback" ]
# for i in myitem:
#     if i[0]== "t":
#       print(i)
#     elif i[0] == "T":
#        print(i)

#create a program that checks the greater among a numbers when a user enter three numbers
# x=int(input("enter your first number:  \n"))
# y=int(input("Enter your second number:  \n"))
# z= int(input("enter your third number: \n"))
# d= int(input("Enter your fourth number \n"))
# #sumnum=(f' {x} {y} {z}')
# # print(sumnum)
# if (x > y) and (x > z) and (x > d) :
#     print(x)
# elif (y > x) and (y > z) and (y > d):
#     print(y)
# elif (z > x) and (z > y) and (z > d) :
#     print(z)  
# elif (d > x) and (d > y) and (d > z):
#     print(d)
# else:
#     print("none")

# def num1(x):
#     return x + 10
# print(num1(20))

# num= lambda a: a + 10
# print(num(10))

# def mydouble(x):
#     return lambda a :a * x **2
# mytunde= mydouble(10)
# print(mytunde(2))


