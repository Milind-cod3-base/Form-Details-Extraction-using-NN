
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'],gpu = False) # load once only in memory.

image_file_name ='lol.png'
image = cv2.imread(image_file_name)

# sharp the edges or image.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
r_easy_ocr=reader.readtext(thresh,detail=0)




"""
    This module contains extra codes which could be
    used later.
"""



#readTyped('thesis_form.pdf')
# code for csv output

# with open('output.csv','w') as output:
#     writer = csv.writer(output)
#     for key, value in readTyped('thesis_form.pdf').items():
#         writer.writerow([key, value])


#readTyped('thesis_form.pdf')


"""
a new function to read the handwritten
it will first read the handwritting using NN
after that, NN will give a number(s) or string 

"""




# below code is helpful when the image is in the form of scanned photo
"""
import pdf2image
import pytesseract
from pytesseract import Output, TesseractError

pdf_path = "document.pdf"

images = pdf2image.convert_from_path(pdf_path)

pil_im = images[0] # assuming that we're interested in the first page only

ocr_dict = pytesseract.image_to_data(pil_im, lang='eng', output_type=Output.DICT)
# ocr_dict now holds all the OCR info including text and location on the image

text = " ".join(ocr_dict['text'])



"""
