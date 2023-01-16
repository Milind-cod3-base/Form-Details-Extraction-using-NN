"""
This module detects the field entry spaces for the Neural network
to take it is an input.

Make a function which gives the read out regions to the neural network,
in a sequential ways, and store the output into the csv file in the 
sequence.

Question: How to store the output of the neural network? and store it inside the csv
"""

import cv2
import tempData
#from google.colab.patches import cv2_imshow

# different parameters
name = 600, 250, 760, 80
matriculation_number = 300, 350, 300, 80
semester = 740, 350, 150, 80
mobile = 1070, 350, 390, 80
address = 270, 430, 1200, 87
first_examiner= 270, 530, 1200, 87
company_name = 130, 750, 1200, 80
supervisor_name = 530, 820, 820, 80
phone = 230, 910, 1200, 70
thesis_title= 130, 1035, 1225, 150
topic_summary =130, 1230, 1225, 130 


# # 2 forms 
# form_d = '/content/dharmes.png'
# form_j = '/content/jomon.png'

# read the input form
#input_form = tempData.read()
input_form = 'D:\projecct\Form-Details-Extraction-using-NN\Scanned_Forms\dharmes.png'

# this function gives the exact location of the data entry
# by the user next to its field
def crop_save(field, form, save_path):
  p1, p2, a1, a2 = field
  image = cv2.imread(form)
  #image_copy = image.copy()
  #image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
  x1, y1 = p1, p2
  x2, y2 = x1 + a1, y1 + a2

  cropped_image = image[y1:y2, x1:x2]
  cv2.imwrite(save_path, cropped_image)