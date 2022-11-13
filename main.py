from fileinput import filename
from tkinter import W
from PyPDF2 import PdfReader
# import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# object to store uploaded file name
# this will be changed as the code has to detect
# the uploaded file and find the name of the file.
fileName = "thesis_form.pdf"



# this only reads the typed inputs
# returns dictionary of categories and entries
def readTyped(file):

    f = PdfReader(file)
    ff = f.get_form_text_fields()

    return ff
    #print(ff)

# stores entry data into a single row of excel file
def storeExcel(file):
    # using pandas converting it directly into xlsx format

    # convert into dataframe
    df = pd.DataFrame(data= readTyped(file), index= [1])

    # convert into csv file with appending mode
    df.to_csv('out1.csv',index= False, mode = 'a')

    # convert into xlsx
    #df.to_excel('out.xlsx', index=False)


# to find the empty / unfilled boxes
# create an empty list
# later we can give a pop about these lists that these entries are empty
def findEmpty(file):
    emptyPlaces = []

    for categories, detail in readTyped(file).items():
        # if detail is empty
        if detail == None:
            emptyPlaces.append(categories)
    
    # returns the list of unfilled categories
    #return emptyPlaces
    print (emptyPlaces)

# function to send unfilled stats
def report(file):
    # total entry spaces in form
    total = len(readTyped(file))

    
    # number of empty entries
    frequency = Counter(readTyped(file).values())
    if None in frequency.keys():

        blank= frequency[None]

    else:
        blank = 0

    filled = total - blank
    #percent_filled = ((total - blank)/ total)*100
    stats = [filled, blank]
    
    # can return multiple values
    return stats

    #print(percent_filled)

# function to print graph of filled and unfilled
def graph(file):
    label = ['Filled', 'Blank']
    plt.pie(report(file), labels= label)
    plt.title('Form Statistics')
    plt.show()



storeExcel(fileName)
#findEmpty(fileName)
#report(fileName)
#readTyped(fileName)
#graph(fileName)