from datetime import datetime
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,jsonify,request,send_file
import json
import io
import os
from io import BytesIO
from dotenv import load_dotenv

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
# set env for secret key
load_dotenv()

secret_id = os.getenv('AI_SERVICE_SECRET_KEY')

# load models

net = cv2.dnn.readNetFromCaffe('models/colorization_deploy_v2.prototxt','models/colorization_release_v2.caffemodel')
classification_model = load_model('models/vgg16_v2_76.h5')

# load reports
img_1 = Image.open('reports/corona_report_00001.jpg')
img_2 = Image.open('reports/corona_report_00002.jpg')
img_3 = Image.open('reports/corona_report_00003.jpg')

def prediction(Image_name):
    # predict image
    resize_image = cv2.resize(Image_name, (224,224)) # convert image size
    test_image = image.img_to_array(resize_image)
    test_image=test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    result = classification_model.predict(test_image) # predict image through VGG16 model
    
    # get category
    cate=''
    max_value = max(list(result[0]))
    max_index = list(result[0]).index(max_value)
    if max_index==0:
        cate='covid'
    elif max_index==1:
        cate='normal'
    else:
        cate='viral'
    # return category
    return cate

# convert X-ray image to color image
def color_scan(image):
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = np.load('models/pts_in_hull.npy')
    # with fs.open('ct-scan_classification/pts_in_hull.npy', 'rb') as file3:
    #     pts = np.load(file3)
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # load the input image from disk, scale the pixel intensities to the
    # range [0, 1], and then convert the image from the BGR to Lab color
    # space
    #image = cv2.imread("img.jpeg")
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # resize the Lab image to 224x224 (the dimensions the colorization
    # network accepts), split channels, extract the 'L' channel, and then
    # perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # pass the L channel through the network which will *predict* the 'a'
    # and 'b' channel values
    'print("[INFO] colorizing image...")'

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize the predicted 'ab' volume to the same dimensions as our
    # input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the
    # predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # convert the output image from the Lab color space to RGB, then
    # clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned
    # 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    return colorized

def check_for_secret_id(request_data):    
    try:
        if 'secret_id' not in request_data.keys():
            return False, "Secret Key Not Found."
        
        else:
            if request_data['secret_id'] == secret_id:
                return True, "Secret Key Matched"
            else:
                return False, "Secret Key Does Not Match. Incorrect Key."
    except Exception as e:
        message = "Error while checking secret id: " + str(e)
        return False,message

@app.route('/Covid_CT-Scan_predictor',methods=['POST'])  #main function
def main():
    key = request.form['secret_id']
    request_data = {'secret_id' : key}
    secret_id_status,secret_id_message = check_for_secret_id(request_data)
    print ("Secret ID Check: ", secret_id_status,secret_id_message)
    if not secret_id_status:
        return jsonify({'message':"Secret Key Does Not Match. Incorrect Key.",
                        'success':False}) 
    else:
        img_params =request.files['image'].read()
        npimg = np.fromstring(img_params, np.uint8)
        #load image
        Image_name = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        prd_result = prediction(Image_name) # get category of image
        date = datetime.today().strftime('%d-%m-%Y') # get live date

        if prd_result=='covid':
            cate = img_1
        elif prd_result=='normal':  
            cate = img_2
        else:
            cate = img_3

        color_coverted = cv2.cvtColor(Image_name, cv2.COLOR_BGR2RGB) #  cv2 to PIL  object
        img2=Image.fromarray(color_coverted)

        # XT scan image
        img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR) # PIL to cv2 object
        img2=color_scan(img2) # color the image

        color_coverted = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) #  cv2 to PIL  object
        pil_image=Image.fromarray(color_coverted)

        # change size of image
        img2 = pil_image.resize((round(pil_image.size[0]*4), round(pil_image.size[1]*4)))

        # overlap image
        cate.paste(img2, (220,1050))

        # set font and text
        # myFont = ImageFont.truetype("arial.ttf", 50)
        
        I1 = ImageDraw.Draw(cate)
        # Add Text to an image
        # I1.text((1160, 410), date, fill=(0, 0, 0))
        
        
        cate = cv2.cvtColor(np.array(cate), cv2.COLOR_RGB2BGR) # PIL to cv2 object
        cate=color_scan(cate) # color the image
        
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (1160, 448)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (0, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        cate = cv2.putText(cate, date, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        I = cv2.cvtColor(cate, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(I.astype('uint8'))
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')
        file_object.seek(0)

        output = send_file(file_object, mimetype='image/PNG') 
    return output

if __name__ == '__main__':
    app.run()