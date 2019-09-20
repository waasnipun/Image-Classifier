import os
from uuid import uuid4
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing import image
from flask import Flask, request, render_template, send_from_directory

__author__ = 'waas'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    #trained model of the dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    new_model = keras.models.load_model('Trained_model.h5')
    #new_model.summary()
    y_binary = to_categorical(y_test)
    loss,acc = new_model.evaluate(x_test,y_binary)
    print("Accuracy of the saved model: {:5.2f}%".format(100*acc))
    class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    print(filename)
    path = "C:/Users/Nipun/Documents/My Projects/TensorFlow Projects/ImageClassifierDisplay/src/images/"+filename
    img = image.load_img(path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classified_image = ""
    images = np.vstack([x])
    classes = new_model.predict(images, batch_size=10)
    print(classes)
    for i in range(len(class_names)):
        if classes[0][i]== 1:
            classified_image =  class_names[i]
            break
    f = open("C:/Users/Nipun/Documents/My Projects/TensorFlow Projects/ImageClassifierDisplay/src/templates/complete_display_image.html",'w')
    message = ("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    
</head>
<body>
<br><img src=" {{url_for('send_image', filename=image_name)}}" height="300" width="300">\n
<br><b>%s</b>
<br>
<br><form id="upload-form" action="{{ url_for('upload') }}" method="POST" enctype="multipart/form-data">

    <strong>Files:</strong><br>
    <input id="file-picker" type="file" name="file" accept="image/*" multiple>
    <div id="msg"></div>
    <input type="submit" value="Upload!" id="upload-button">
</form>
</body>
<<script>

    $("#file-picker").change(function(){

        var input = document.getElementById('file-picker');
        
        for (var i=0; i<input.files.length; i++)
        {
        //koala.jpg, koala.JPG substring(index) lastIndexOf('a') koala.1.jpg
            var ext= input.files[i].name.substring(input.files[i].name.lastIndexOf('.')+1).toLowerCase()
            var filename = input.files[i].name.substring()
            if ((ext == 'jpg') || (ext == 'png'))
            {
                $("#msg").text("Files are supported")
            }
            else
            {
                $("#msg").text("Files are NOT supported")
                document.getElementById("file-picker").value ="";
            }

        }


    } );

</script>
</html>"""%classified_image)

    f.write(message)
    f.close()
    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete_display_image.html", image_name=filename)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
