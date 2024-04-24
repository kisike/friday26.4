# # first_name=input("enter your first name: ")
# # last_name=input("enter your last name: ")
# # print("my name is"+ ' '+first_name+ ' '+last_name)

# # print(type(num1))

# # casting is used to add integers together.

# # concartination is used to add string values.

# # print(type{int(num)})

# # address = '40, University Road, Off Heberty Macauly Road, Akoka Yaba, Lagos State Nigeria'
# # # print(len(address))
# # print(address[0:40:2])
# # print(address.capitalize())
# # print(address.count("a"))
# # print(address.isdecimal())
# # new_address=address.split()
# # print(type(new_address))
# # new_address.reverse()
# # print(new_address)
# # print(address.endswith("a"))
# # print(address.startswith("b"))

# colors=['blue', 'red', 'yellow', 'purple']
# mycolor=['cream']
# # colors.pop(2)
# # colors.append('green')
# # colors.extend(mycolor)
# # colors.reverse()
# # colors.reverse('blue')
# # color=colors.count('blue')
# # new_color = colors.copy()
# # colors.sort()
# colors.insert(0, 'pink')
# print(colors)


# try:
#     tunde = open("tunde.txt", "x")
#     try:
#         tunde.write("Welcome to digital fortress")
#     except:
#         print("Something went wrong with the creation of thr file")
#     else:
#         print("file created successfully")
# except NameError:
#      print("something went wrong with the file")
# finally:
#        tunde.close()
# # print("The logic is done running")

# tunde =10
# try:
#     print(tunde)
# except:
#     print("tunde is not defined")
# else:
#     print("Nothing went wrong")


tunde = open("tunde.txt", "x")
print(tunde.write("Welcome to digital fortress"))
      