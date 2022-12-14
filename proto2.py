# this module is to test OCR
# now the testing begins here. We need to work here. 
import pdf2image
import pytesseract
from pytesseract import Output, TesseractError

pdf_path = "tobi1.pdf"

images = pdf2image.convert_from_path(pdf_path)

pil_im = images[0] # assuming that we're interested in the first page only

ocr_dict = pytesseract.image_to_data(pil_im, lang='eng', output_type=Output.DICT)
# ocr_dict now holds all the OCR info including text and location on the image

text = " ".join(ocr_dict['text'])


print(text)



# from PyPDF2 import PdfReader

# reader  = PdfReader('tobi1.pdf')
# page = reader.pages[0]

# print(reader.getFormTextFields())
