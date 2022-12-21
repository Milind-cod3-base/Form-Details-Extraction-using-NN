from fileinput import filename
#from tkinter import W
from PyPDF2 import PdfReader
# import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

from tkinter.messagebox import showinfo

from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.lang import Builder
from kivy.uix.image import Image

# object to store uploaded file name
# this will be changed as the code has to detect
# the uploaded file and find the name of the file.
fileName = "thesis_form.pdf"

class MainWindow(Screen):
    pass

class StatsWindow(Screen):
    pass
        

class DetailsWindow(Screen):
    def show(self):
        return findEmpty(fileName)
        

class ManualWindow(Screen):
    pass

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
    text = "\n \n"

    for categories, detail in readTyped(file).items():
        # if detail is empty
        if detail == None:
            emptyPlaces.append(categories)

    # convert above list into a presentable paragraph
    msg = text.join(emptyPlaces)
    
    # returns the list of unfilled categories
    #return emptyPlaces
    #print (emptyPlaces)

    # a dialog box consisting of empty entries
    #showinfo(title="Empty entries", message= msg)
    return msg
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
    plt.pie(report(file), labels= label, autopct= '%1.0f%%')
    plt.title('Form Statistics')
    #plt.show()
    plt.savefig('D:\projecct\Form-Details-Extraction-using-NN\output_graph.png')


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
    
    MyApp().run()
    graph(fileName)
    #print(findEmpty(fileName))
    
