from fileinput import filename
#from tkinter import W
from PyPDF2 import PdfReader
import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

from tkinter.messagebox import showinfo

from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.lang import Builder
from kivy.uix.image import Image
from kivymd.uix.filemanager import MDFileManager

import cv2

import tempData
import fieldDetect

# object to store uploaded file name
# this will be changed as the code has to detect
# the uploaded file and find the name of the file.
fileName = tempData.read()



class MainWindow(Screen, MDApp):

    
    # constructor
    def __init__(self, **kw):
        super().__init__(**kw)
        self.file_manager_obj = MDFileManager(
            select_path= self.select_path ,   # method of which window will open first
            exit_manager= self.exit_manager, # method to exit the file manager
            preview=True

        )

    # method to open file manager
    def open_file_manager(self):
        # opening file manager
        #self.file_manager_obj_main.show('/')
        self.file_manager_obj.show("D:\projecct")
    
    
    # gets the path of the file
    def select_path(self, path):
        # global fileName

        # fileName = path
        
        tempData.save(path)
        
        #print(path)
        
        self.exit_manager_main()
        
    """This stays, important for filemanager to close"""
    def exit_manager(self):
        self.file_manager_obj.close()

class StatsWindow(Screen):
    pass
        

class DetailsWindow(Screen):
    def show(self):
        print(findEmpty())
        

class ManualWindow(Screen):
    pass







# this only reads the typed inputs
# returns dictionary of categories and entries
# def readTyped(file):

#     f = PdfReader(file)
#     ff = f.get_form_text_fields()

#     return ff
    #print(ff)

# stores entry data into a single row of excel file
def storeExcel():

    input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11 = fieldDetect.outPutAI()
    
    with open('out1.csv', 'a', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        
        # Write the 11 inputs as a row in the CSV file
        writer.writerow([input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11])
    
    
    
    
    
    
    
    
    
    
    # # using pandas converting it directly into xlsx format

    # # convert into dataframe
    # df = pd.DataFrame(data= readTyped(file), index= [1])

    # # convert into csv file with appending mode
    # df.to_csv('out1.csv',index= False, mode = 'a')

    # # convert into xlsx
    # #df.to_excel('out.xlsx', index=False)


# to find the empty / unfilled boxes
# create an empty list
# later we can give a pop about these lists that these entries are empty
# def findEmpty(file):
#     emptyPlaces = []
#     text = "\n \n"

#     for categories, detail in readTyped(file).items():
#         # if detail is empty
#         if detail == None:
#             emptyPlaces.append(categories)

#     # convert above list into a presentable paragraph
#     msg = text.join(emptyPlaces)
    
#     # returns the list of unfilled categories
#     #return emptyPlaces
#     #print (emptyPlaces)

#     # a dialog box consisting of empty entries
#     #showinfo(title="Empty entries", message= msg)
#     return msg

def findEmpty():

    file_path = 'D:\projecct\Form-Details-Extraction-using-NN\output1.csv'
    
    with open(file_path, 'r') as csvfile:
        # use the DictReader to read the CSV file
        reader = csv.DictReader(csvfile)
        # get the fieldnames (column headers)
        headers = reader.fieldnames

        # create a set to store the empty columns
        empty_columns = set()

        # iterate through the rows
        for row in reader:
            # iterate through the headers
            for header in headers:
                # check if the value for the current header is empty
                if not row[header]:
                    # add the header to the set of empty columns
                    empty_columns.add(header)
        
        # convert set into tuple
        empty_columns = tuple(empty_columns)

        # converting tuple into a line spaced str
        empty_columns = str('\n'.join(empty_columns))
        # print the empty columns
        return empty_columns

# function to send unfilled stats
# def report(file):
#     # total entry spaces in form
#     total = len(readTyped(file))

    
#     # number of empty entries
#     frequency = Counter(readTyped(file).values())
#     if None in frequency.keys():

#         blank= frequency[None]

#     else:
#         blank = 0

#     filled = total - blank
#     #percent_filled = ((total - blank)/ total)*100
#     stats = [filled, blank]
    
#     # can return multiple values
#     return stats

#     #print(percent_filled)

# function to print graph of filled and unfilled
def graph(file_name, row_index=2):
    
    filled_count = 0
    empty_count = 0
    # Open the CSV file
    with open(file_name, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Get the specified row
        for i, row in enumerate(reader):
            if i == row_index:
                for item in row:
                    if item:
                        filled_count += 1
                    else:
                        empty_count += 1
                break
    # Create a pie chart to show the proportion of filled and empty elements
    labels = ['Filled', 'Empty']
    sizes = [filled_count, empty_count]
    colors = ['#ff9999','#66b3ff']
    plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True)
    plt.axis('equal')
    #plt.show()
    plt.savefig('D:\projecct\Form-Details-Extraction-using-NN\output_graph.png')

    # label = ['Filled', 'Blank']
    # plt.pie(report(file), labels= label, autopct= '%1.0f%%')
    # plt.title('Form Statistics')
    # #plt.show()
    # plt.savefig('D:\projecct\Form-Details-Extraction-using-NN\output_graph.png')


#storeExcel(fileName)
#findEmpty(fileName)
#report(fileName)
#readTyped(fileName)
#graph(fileName)

sm =  ScreenManager()

screens = [MainWindow(name="mainW"),
           StatsWindow(name="statsW"),
           DetailsWindow(name="detailW"),
           ManualWindow(name="manualW")
           ]


for i in screens:
    sm.add_widget(i)

    
class MyApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Light"
        return Builder.load_file("my.kv")

if __name__ == "__main__":
    #graph('D:\projecct\Form-Details-Extraction-using-NN\output_graph.png')
    #storeExcel(fileName)
    MyApp().run()
    # print(readTyped(fileName))
    #print(findEmpty(fileName))
    
