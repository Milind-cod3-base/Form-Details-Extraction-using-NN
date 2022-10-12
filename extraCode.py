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
