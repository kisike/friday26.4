# # class Person:
# #     name= "samsung"
# # new_person=  Person() 
# # print(new_person.name)

# class Person:
#     def __init__(self, firstname, lastname, age, origin ):
#         self.firstname= firstname
#         self.lastname = lastname
#         self.age= age 
#         self.origin= origin
#     def __str__(self) :
#         return f'{self.firstname} {self.lastname} {self.age} {self.origin}'
#     def mydescription(self):
#         return f'My name is  {self.firstname} {self.lastname}, i am  {self.age} yerars old, and i am from {self.origin}'

# new_person = Person("tunde", "john", 10, "lagos state") 
# print(new_person.mydescription())


# class Mynew(Person):
#     def __init__(self, firstname, lastname, age, origin, serial_number, color):
#         super().__init__(firstname, lastname, age, origin)
#         self.serial_number= serial_number
#         self.color = color
#     def __str__(self):
#         return f'{self.firstname} {self.lastname} {self.age} {self.origin} {self.serial_number} {self.color} '

# x = Mynew('tunde','emeka', 20, 'ogun state', 11234 ,'color')
# print(x)



# class Car:
#     def __init__(self, name, model,brand, color, ):
#         self.name = name
#         self.model = model
#         self.brand = brand 
#         self.color= color
#     def __str__(self):
#         return f'{self.name} {self.model} {self.brand} {self.color}'
# Cars= Car('Pickup', 2009, 'toyota', 'black')   
# # print(Cars)

# class Mynewcar(Car):
#     def __init__(self, name, model,brand, color, engine, price , use ):
#         super().__init__(name, model,brand, color)
#         self.engine= engine
#         self.price= price
#         self.use= use
#     def __str__(self):
#         return f'{self.name}, {self.model},{self.brand}, {self.color}, {self.engine}, {self.price},{self.use} '
# x= Mynewcar('benz',2018,'benzo','black', 1234, 2000000, 'foreign used')
# print(x)



# class person:
#     def __iter__(self):
#         self.a= 1
#         return self
#     def __next__(self):
#         x = self.a 
#         self.a += 1
#         return x
# num= person()
# num2= iter(num)
# print(next(num2))


# fruits= ("banana", "watermelon", "grape")
# myfruits= iter(fruits)
# print(next(myfruits))

# polymorphism
# class car:
#     def __init__(self, brand, name):
#           self.brand= brand
#           self.name= name
#     def tunde(self):
#      return f'The name of my car is {self.name} and the brand is {self.brand}'

# class school:
#     def __init__(self, brand, name):
#           self.brand= brand
#           self.name= name
#     def tunde(self):
#      return f'The name of my school is {self.name} and the brand is {self.brand}'
    
# class Phone:
#     def __init__(self, brand, name):
#           self.brand= brand
#           self.name= name
#     def tunde(self):
#      return f'The name of my phone is {self.name} and the brand is {self.brand}'
    
# mycar= car("toyota", "camry")
# myphone= Phone("samsung", "fold")
# myschool= school("unilag", "first choice")
# for i in (mycar, myphone, myschool):
#     print(i.tunde())


# hotel name , location , capacity, state and country
# class first_Hotel:
#     def __init__(self, name , location, capacity, state, country):
#         self.name= name
#         self.location= location
#         self.capacity= capacity
#         self.state= state
#         self.country= country
#     def daniel(self):
#         return(f'The hotel name is {self.name} and it is located at  {self.location}, with capacity of  {self.capacity} in {self.state}, {self.country}')

# class second_Hotel:
#     def __init__(self, name , location, capacity, state, country):
#         self.name= name
#         self.location= location
#         self.capacity= capacity
#         self.state= state
#         self.country= country
#     def daniel(self):
#         return(f'The hotel name is {self.name} and it is located at  {self.location}, with capacity of  {self.capacity} in {self.state}, {self.country}')


# class third_Hotel:
#     def __init__(self, name , location, capacity, state, country):
#         self.name= name
#         self.location= location
#         self.capacity= capacity
#         self.state= state
#         self.country= country
#     def daniel(self):
#         return(f'The hotel name is {self.name} and it is located at  {self.location}, with capacity of  {self.capacity} in {self.state}, {self.country}')


# Hotel_1 = first_Hotel("Havanah", "Gowown estate", "300 rooms", "Lagos state","Nigeria")
# Hotel_2 = second_Hotel("Exclusite", "Egbeda", "203 rooms", "ogun state", "Nigeria")
# Hotel_3 = third_Hotel("Arabian", "Agbesan", "400 rooms", "ondo state", "Nigeria")
# for i in (Hotel_1,Hotel_2,Hotel_3):
#     print(i.daniel())

#scope: is setting of function inside a function
# def myname():
#     tunde= 10
#     print(tunde)
# myname()
# x= 20
# def myname():
#     tunde= 10
#     global x

#     def mytunde():
#         y= 15
#         print(tunde + x + y)
#     mytunde()
# myname()


# import dan
# x= dan.mysum(4, 6)
# print(x)

# y= dan.employees


# import math
# x = abs(10.25) 
# print(x)
# x = pow(4,5)
# print(x)
# x = math.ceil(4.3)
# print(x)
# x= math.floor(4.5)
# print(x)
# x = math.sqrt(100)
# print(x)


# import random
# x= random.randint(10,50)
# print(x)


# import datetime
# x = datetime.datetime.now()
# print(x)
# print(x.year)
# print(x.strftime('%A'))
# print(x.strftime('%a'))
# print(x.strftime('%W'))
# print(x.strftime('%w'))
# print(x.strftime('%D'))
# print(x.strftime('%b'))
# print(x.strftime('%M'))
# print(x.strftime('%f'))

# tunde =10
# try:
#     print(tunde)
# except:
#     print("tunde is not defined")
# else:
#     print("Nothing went wrong")

try:
    tunde = open("tunde.txt", "x")
    try:
        tunde.write("Welcome to digital fortress")
    except:
        print("Something went wrong with the creation of thr file")
    else:
        print("file created successfully")
except NameError:
    print("something went wrong with the file")
finally:
    print("The logic is done running")















