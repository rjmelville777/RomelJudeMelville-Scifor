# -*- coding: utf-8 -*-
"""Use case of inheritance.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uoW_nTbCxWiITxGLRQk2GUo2cjpHqDjq
"""

class Bank:
    def __init__(self,customer_name,customer_pin,bank_balance):
        self.customer_name = customer_name
        self.customer_pin = customer_pin
        self.bank_balance = bank_balance

    def withdraw(self,amount):
        if amount <= 0:
            return "Invalid amount"
        if amount > self.bank_balance:
            return "Insufficient balance"
        self.bank_balance -= amount
        return f"Withdraw {amount} rupees. Remaining balance:{self.bank_balance} rupees"

    def show_balance(self):
        return f"Current balance for {self.customer_name}:{self.bank_balance} rupees"

    def deposit(self,amount):
        if amount <=0:
            return "Invalid amount"

        self.bank_balance += amount
        return f"Deposited{amount} rupees.New balance:{self.bank_balance} rupees"

customer_1 = Bank("Romel","5389",10000)
deposit_amount = customer_1.deposit(2500)
print(deposit_amount)
balance_info = customer_1.show_balance()
print(balance_info)

