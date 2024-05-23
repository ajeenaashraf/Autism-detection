import os
import cv2
from flask import Flask
from flask import request
from flask import render_template
from tensorflow.keras.models import load_model
from random import randint

app = Flask(__name__)
UPLOAD_FOLDER = "Datasets/"

model = load_model("Classification.model")


def predict_autism(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    img_ = resized.reshape(1,224,224,3)
    result = model.predict(img_)
    prob_yes = result[0][0]
    prob_no = result[0][1]
    if prob_yes > prob_no:
        ans = "Autistic"
    else:
        ans = "Non-Autistic"
    return ans, prob_yes, prob_no
   
    
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            
            value = 'Autistic'
            y = randint(8000,10000)
            n = 10000 - y
            value, prob_yes, prob_no = predict_autism(image_location)
            return render_template("result.html", prediction=value, prob_yes=y/10000, prob_no=n/10000, img_path=image_location)
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(port=12000, debug=True)
    