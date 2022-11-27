from flask import Flask , jsonify , request
from classifier import get_prediction
import cv2
import numpy as np

app = Flask (__name__)

@app.route("/predict-digit" , method = ["POST"])
def perdict_data():
  #  image = cv2.imdecode(np.fromstring(request.files.get("digit").read(),np.uint8),cv2.IMREAD_UNCHANGED)
    image = request.files.get("digit")
    prediction = get_prediction(image)
    return jsonify({
        "prediction" : prediction
    }), 202

if __name__ == "__main__":
    app.run(debug=True)