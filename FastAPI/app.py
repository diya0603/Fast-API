# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from cattle import Cattle
import numpy as np
import pickle
import pandas as pd
import joblib
# 2. Create the app object
app = FastAPI()
#pickle_in = open("classifier.pkl","rb")
classifier=model = joblib.load('voting_clf_model.pkl')

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_cattle(data:Cattle):
    data = data.dict()
    body_temperature=data['body_temperature']
    heart_rate=data['heart_rate']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[body_temperature,heart_rate]])
    if(prediction[0] > 0.5):
        prediction="Unhealthy"
    else:
        prediction="Healthy"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload   