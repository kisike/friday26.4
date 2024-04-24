# secret_number= 20
# guess_count = 0
# guess_limit= 3
# while guess_count < guess_limit:
#     guess= int(input("Enter your guess: "))
#     guess_count += 1
#     if guess == secret_number:
#         print(" You won!")
#         break
#     elif guess is not secret_number:
#         print("Try again")
# else:
#     print("Sorry you failed!")

# CAR GAME 
Command= ""
started= False
# while Command.lower() != "quit":
while True:
    command= input(">").lower()
    if command == "start":
        if started:
            print("Car already started")
        else:
            started = True
        print("Car started...")
    elif command == "stop":
        if not started:
            print("Car is already stopped!")
        else:
            started= False
        print("Car stopped")
    elif command == "help":
        print("""
start- start the car
stop- stop the car
quit- to quit 
""")
    elif command == "quit":
        break
    else:
        print("Sorry i don't understand that")
