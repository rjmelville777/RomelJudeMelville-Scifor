#!/usr/bin/env python
# coding: utf-8

# # INTEGERS

# In[77]:


x = 5
y = 10
z = 0


# In[78]:


sum_result = x + y  
sum_result


# In[79]:


product_result = x * z  
product_result


# In[57]:


division_result = y / x  
division_result


# In[81]:


quotient = x // 2 
quotient


# In[82]:


remainder = x % 2 
remainder


# # FLOAT

# In[61]:


x = 3.14
y = -0.5
z = 1.0


# In[62]:


sum_result = x + y  
sum_result


# In[63]:


product_result = x * z  
product_result


# In[64]:


division_result = y / 2  
division_result


# # STRING

# In[65]:


a = 'Hello, World!'
b = "Python is awesome!"


# In[66]:


#Combining two strings
c = a + " " + b
c


# In[67]:


# Repetition
repeated_string = a * 3
repeated_string


# In[68]:


#Accessing Characters
indexing = a[0]  
indexing


# In[70]:


#String Slicing
slicing = a[7:12] 
slicing


# In[71]:


lower = a.lower()
lower


# In[72]:


upper = b.upper()
upper


# In[73]:


length = len(b)
length


# # LIST

# In[83]:


#Creating a List
my_list = [1, 2, 3, 4, 5]
my_list


# In[84]:


#Accessing Elements
first_element = my_list[0]  # Access the first element
first_element


# In[85]:


second_element = my_list[1]  # Access the second element
second_element


# In[86]:


#Slicing
slicing = my_list[1:4]  # Extract elements from index 1 to 3
slicing


# In[87]:


#Adding Elements
my_list.append(6)  # Adds 6 to the end of the list
my_list


# In[88]:


my_list.insert(1,5)  # Inserts 5 at index 1
my_list


# In[90]:


#Removing Elements
my_list.remove(5)  # Removes the first occurrence of 5
my_list


# In[91]:


p = my_list.pop(2)# Removes and returns the element at index 2
p


# In[92]:


#Length of a list
length = len(my_list)
length


# # TUPLE

# In[19]:


#Creating a Tuple
my_tuple = (1, 2, 3, 'a', 'b', 'c')
my_tuple


# In[20]:


#Accessing Elements
first_element = my_tuple[0]  # Access the first element
first_element


# In[21]:


second_element = my_tuple[1]  # Access the second element
second_element


# In[104]:


#Addition
tup_1 = (1,2,3,4,5)
tup_2 = (6,7,8,9,1)
add = tup_1 + tup_2
add


# In[23]:


#Slicing
subset = my_tuple[2:5]  # It extracts elements from index 2 to 4
subset


# # DICTIONARY

# In[95]:


#Creating a Dictionary
my_dict = {'Name': 'Jude', 'Age': 25, 'City': 'New York'}
my_dict


# In[97]:


#Accessing Values
name = my_dict['Name']  
name
#You can access the values in a dictionary by referencing the associated key


# In[98]:


age = my_dict['Age'] 
age


# In[100]:


#Modifying Values
my_dict['Age'] = 26  
my_dict


# In[101]:


#Adding New Key-Value Pairs
my_dict['Occupation'] = 'Doctor'
my_dict


# In[103]:


#Removing Key-Value Pairs
del my_dict['City']
my_dict


# In[41]:


#Dictionary Methods
#1. keys(): Returns a list of all the keys in the dictionary
keys_list = my_dict.keys()
keys_list


# In[42]:


#2. values(): Returns a list of all the values in the dictionary
values_list = my_dict.values()
values_list


# In[43]:


#3. items(): Returns a list of key-value pairs as tuples
items_list = my_dict.items()
items_list


# # SET

# In[44]:


#Creating a Set
my_set = {1, 2, 3, 4, 5}
my_set


# In[45]:


#Adding and Removing Elements
my_set.add(6)
my_set


# In[46]:


my_set.remove(3)
my_set


# In[48]:


#Set Methods
#1. union() returns a new set with all elements from the sets
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union_set = set1.union(set2)
union_set


# In[49]:


#2. intersection() returns a new set with common elements
intersection_set = set1.intersection(set2)
intersection_set


# In[50]:


#3. difference() returns a new set with elements in the first set but not in the second
difference_set = set1.difference(set2)
difference_set

