import cv2
import streamlit as st
import tensorflow as tf
import os
import numpy as np
import PIL
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 
from sklearn.metrics import mean_absolute_error, mean_squared_error
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
## Page Title
#st.set_page_config(page_title = "Cats vs Dogs Image Classification")
st.title("POTHOLE DETECTION")
st.markdown("---")
#st.caption('HOSTED BY CHAITANYA 201801330017')

model_path='pothole.tflite'
st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
F1 = st.image([])
F2 = st.image([])
F3 = st.image([])
F4 = st.image([])


while run:
    camera = cv2.VideoCapture("8.mp4")
    _, frame1 =camera.read()
    _, frame2= camera.read()
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    
    FRAME_WINDOW.image(frame1)
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, threshold1=30, threshold2=150)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        pothole = area
        if cv2.contourArea(contour) < 10000:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0 ,255, 0), 2)
        cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        print('pothole',round(pothole,2))
        cv2.putText(frame1, f'pothole:{pothole}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #cv2.putText(frame1, "Status : {}".format('potholes'), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
        #print("Number of contours in image:",len(contours))
        #st.image("dilated", dilated)
        F4.image(dilated)   
    F1.image(frame1)
    F2.image(thresh)
    F3.image(edges)
    frame1=frame2
    ret, frame2=camera.read()
    
df = pd.read_csv("mlppa.csv")

y = df['Impulse'].values.reshape(-1, 1)
X = df['Area'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
def calc(slope, intercept, Area):
    return slope*Area+intercept

score = calc(regressor.coef_, regressor.intercept_, 9.5)

score = regressor.predict([[9.5]])

y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')



# Load the labels into a list
classes = ['pothole', 'road_with_cracks','pothole_with_water','pothle with water','pothole with water','pothole_with_ water','pothole _with_ water','pohole with water','group of potholes','group_of_potholes','Marked_Speedbreaker','Unmarked_Speedbreaker','cracks']
#label_map = model.model_spec.config.label_map
#for label_id, label_name in label_map.as_dict().items():
 # classes[label_id-1] = label_name

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  scores = get_output_tensor(interpreter, 0)
  boxes = get_output_tensor(interpreter, 1)
  count = int(get_output_tensor(interpreter, 2))
  classes = get_output_tensor(interpreter, 3)

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
      
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.3):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path, 
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)
  
  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute 
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])
    w=xmax-xmin
    h=ymax-ymin
    # Find the class index of the current object
    class_id = int(obj['class_id'])
    st.write(classes[class_id])
    #st.write(classes)
    #st.write(class_id)
   

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8


## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])
if uploaded_file is not None:
    with open(os.path.join("/tmp",uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())
    path = os.path.join("/tmp",uploaded_file.name)
    URL =path
    DETECTION_THRESHOLD = 0.3

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Run inference and draw detection result on the local copy of the original file
    detection_result_image = run_odt_and_draw_results(
        URL, 
        interpreter, 
        threshold=DETECTION_THRESHOLD
    )

    # Show the detection resulthgf
    st.image(detection_result_image)
    
    

 













