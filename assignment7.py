#write a program to create a guessing game using the random amd randint
# import random
# lower_bound= 1
# upper_bound = 30
# guess_count = 0
# guess_limit = 3
# secret_number= 30
# while guess_count < guess_limit:
#         guess= input(random.randint(lower_bound, upper_bound))
#         guess_count += 1
#         if guess == secret_number:
#                 print("You won!")
#         else:
#             print("Try again")
#         break


# MRtunde game
# import random
# def guessing_game():
#     secret_number = random.randint(0,5)
#     print(secret_number)
#     my_name = input("Enter your name:  \n")
#     print(f'Mr {my_name} Welcome to baba ijebu enterpise betting platform')
#     attempt = 0
#     while True:
#         guess= int(input("Enter your guess number   \n"))
#         attempt += 1
#         if guess > secret_number:
#             print("Your guess is too high try again")
#         elif guess < secret_number:
#             print("Your guess is low try again")
#         else:
#             print(f"congratulations! You have just won {secret_number} after {attempt} attempts")
    


# guessing_game()
    

#ATM project
# class BankAccount:
#    starting_balance = 0.00 # USD
 
#    def _init_(self, first_name, last_name):
#        self.first_name = first_name
#        self.last_name = last_name
 
#    def deposit(starting_balance, amount):
#        balance = int(starting_balance + amount)
#        return balance
  
#    def withdraw(self, balance, amount):
#        try:
#            balance = balance - amount
#        except balance <= 0.00:
#            raise ValueError('Transaction declined. Insufficient funds. Deposit some money first.')
#            withdraw(self,balance, amount)
#        else:
#            return balance
# D= BankAccount.deposit(200,400)
# print(D())


# user_name= input("Enter your name:  \n")
# bank_greetings= ("welcome") + (" to union bank")
# greetings = bank_greetings + (" ") + user_name
# print(greetings)
# class BankAccount:
#    starting_balance = 0.00 # USD
 
#    def _init_(self, first_name, last_name):
#        self.first_name = first_name
#        self.last_name = last_name
 
#    def deposit(starting_balance, amount):
#        balance = int(starting_balance + amount)
#        return balance
  
#    def withdraw(self, balance, amount):
#        try:
#            balance = balance - amount
#        except balance <= 0.00:
#            raise ValueError('Transaction declined. Insufficient funds. Deposit some money first.')
#            withdraw(self,balance, amount)
#        else:
#            return balance
# D= BankAccount.deposit(200,400)
# print(D())

pin= 1234
card_number= 1234567891111
class Atm:
    def __init__(self, name, balance=0):
        self.balance = balance
        self.name = name
    def __str__(self):
        return f"Mr/miss/mrs {self.name} your curent balance is {self.balance}"
    def withdraw (self,amount):
        if amount > self.balance:
            return "insufficient funds"
        else:
            self.balance -= amount
            return f"withdrawal successful and current balance is :{self.balance}"
    def deposite(self, amount):
        self.balance += amount
        return f'Deposite succesful your current balance is : {self.balance}'
    
print("Welcome to Digital fortress Micro finanace Bank")
myname= input("Please Enter your name:  \n")
info = int(input("Enter your card number"))
if info == card_number:
    print(f"Welcome {myname}  \n")
else :
    print("Card does not exist")
    exit()
user_pin = int(input("Please enter your pin"))
if user_pin == pin :
    print("Welcome")
else:
    print("Incorrect pin")
    exit()
user = Atm(myname)
while True:
    choice = input(""""
            what do you like to do?     
            press 1 for deposit
            press 2 for withdraw
            press 3 to check balance
            press 4 to exit   \n
         """) 
    if choice == "3":
        print(user)
    elif choice == "1":
        amount =int(input("Enter your deposit"))
        if amount <= 0:
            print("invalid transaction")
        else:
            print(user.deposite(amount))
    elif choice == "2":
        amount =int(input("Enter your withdrawal amount"))
        if amount <= 0:
            print("invalid transaction")
        else:
            print(user.withdraw(amount))
    if choice == "4":
        print(f"Mr/miss/mrs {myname} thank you for banking with us")
    
    
        

