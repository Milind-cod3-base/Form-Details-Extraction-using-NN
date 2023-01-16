"""
This module detects the field entry spaces for the Neural network
to take it is an input.

Make a function which gives the read out regions to the neural network,
in a sequential ways, and store the output into the csv file in the 
sequence.

Question: How to store the output of the neural network?  and store it inside the csv
"""
import cv2
import tempData
import os
import glob
import numpy as np
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


# need a function to always call NN and input the detail in it
def feedAI():

  # below are the paths of the image cutouts (cropped and resized) which are ready to be fed into NN
  nameCut = crop_image(field=name, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\cname.png')
  matriculation_numberCut = crop_image(field=matriculation_number, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\matriculation_number.png')
  semesterCut = crop_image(field=semester, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\semester.png')
  mobileCut = crop_image(field=mobile, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\mobile.png')
  addressCut = crop_image(field=address, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\caddress.png')
  first_examinerCut = crop_image(field=first_examiner, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\cfirst_examiner.png')
  company_nameCut = crop_image(field=company_name, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\company_name.png')
  supervisor_nameCut = crop_image(field=supervisor_name, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\supervisor.png')
  phoneCut = crop_image(field=phone, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\supervisor_name.png\phone.png')
  thesis_titleCut = crop_image(field=thesis_title, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\cthesis_title.png')
  topic_summaryCut = crop_image(field=topic_summary, form= input_form, save_path='D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\ctopic_summary.png')

  return nameCut, matriculation_numberCut, semesterCut, mobileCut, addressCut, first_examinerCut, company_nameCut, supervisor_nameCut, phoneCut, thesis_titleCut, topic_summaryCut

# a function for NN to take above paths as input returns the outputs and deletes all the files
# in the scanned forms folder

def outPutAI():
  """
  name = runAI() which reads image path as feedAI()[0]
  matriculation = runAI() with feedAI()[1]

  store all of them in variables
  """



  """might not be required as it refreshes"""
  # clean the scanned forms folder before quiting this function for next batch.

  # use glob to get list of all the files in directory
  files = glob.glob('D:\projecct\Form-Details-Extraction-using-NN\Cutoutputs\*')

  for f in files:
    os.remove(f)


  #return name, matri. nu. etc in text which could be stored in csv

"""
def 


"""




# # 2 forms 
# form_d = '/content/dharmes.png'
# form_j = '/content/jomon.png'

# read the input form
input_form = tempData.read()
#input_form = 'D:\projecct\Form-Details-Extraction-using-NN\Scanned_Forms\dharmes.png'

# this function gives the exact location of the data entry
# by the user next to its field, crops that text input
# and saves in the save path. and returns the save path for NN to pick up
def crop_image(field, form, save_path):
  p1, p2, a1, a2 = field
  image = cv2.imread(form)

  target_width = 1024
  target_height = 1028
  
  #image_copy = image.copy()
  #image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
  x1, y1 = p1, p2
  x2, y2 = x1 + a1, y1 + a2

  cropped_image = image[y1:y2, x1:x2]

  img = cropped_image

  # Get the aspect ratio of the image
  aspect_ratio = img.shape[1] / img.shape[0]

  # Calculate the width and height of the resized image
  if aspect_ratio > 1:
      width = target_width
      height = int(width / aspect_ratio)
  else:
      height = target_height
      width = int(height * aspect_ratio)

  # Resize the image
  resized_img = cv2.resize(img, (width, height))

  # Create a black image with the target size
  target_img = np.zeros((target_height, target_width, 3), np.uint8)

  # Calculate the position to place the resized image
  y = int((target_height - height) / 2)
  x = int((target_width - width) / 2)

  # Place the resized image on the black image
  target_img[y:y+height, x:x+width] = resized_img

  # Save the final image
  cv2.imwrite(save_path, target_img)

  return save_path


  


#crop_image(field=name, form= input_form)
#feedAI()
#outPutAI()
#print(feedAI()[0])
#print(feedAI()[1])
