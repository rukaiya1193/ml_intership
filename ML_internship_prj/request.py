import requests

url = 'http://localhost:0000/predict_api'
r = requests.post(url, json={'Reason_for_absence':2, 'Month_of_absence':9, 'Day_of_the_week':6, 'Seasons':3})

print(r.json())