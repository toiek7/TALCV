from label_studio_ml.model import LabelStudioMLBase
import requests, os
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np
import traceback

LS_URL = os.environ['LABEL_STUDIO_BASEURL']
#LS_API_TOKEN = os.environ['LABEL_STUDIO_API_TOKEN']

class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.model = YOLO("best.pt")
        self.labels= list(self.model.names.values())

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: model returns 
            the list of predictions based on input list of tasks 
        """
        print("!!! predict CALLED with tasks:", tasks)
        try:
          task = tasks[0]
  
          predictions = []
          score = 0
          SESSION_ID = os.environ.get('LS_SESSION_ID')
          cookies = {"sessionid": SESSION_ID}
          
          # If task['data']['image'] already starts with 'http', skip LS_URL
          if task['data']['image'].startswith("http"):
              image_url = task['data']['image']
          else:
              image_url = LS_URL + task['data']['image']
          response = requests.get(image_url, cookies=cookies)
          print("Trying to fetch image url:", image_url)
          print("Status code:", response.status_code)
          if response.status_code != 200:
              print("Error! Response returned status:", response.status_code)
              print("Content sample:", response.content[:500])  # print first 500 chars
              # Return an empty prediction dictionary for LS
              return [{
                  "result": [],
                  "score": 0,
                  "model_version": "v8n",
              }]
          
          image = Image.open(BytesIO(response.content))
          original_width, original_height = image.size
          print(image.size)        
          results = self.model.predict(image)
          count=0
          for result in results:
              for i, prediction in enumerate(result.boxes):
                  # Get the class index from YOLO
                  cls_index = int(prediction.cls.item())
                  # Map to the label string
                  label_name = self.labels[cls_index]
                  xyxy = prediction.xyxy[0].tolist()
                  predictions.append({
                      "id": str(i),
                      "from_name": self.from_name,
                      "to_name": self.to_name,
                      "type": "rectanglelabels",
                      "score": prediction.conf.item(),
                      "original_width": original_width,
                      "original_height": original_height,
                      "image_rotation": 0,
                      "value": {
                          "rotation": 0,
                          "x": xyxy[0] / original_width * 100, 
                          "y": xyxy[1] / original_height * 100,
                          "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                          "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                          "rectanglelabels": [label_name]
                      }
                  })
                  score += prediction.conf.item()
                  count+=1
          avg_score = (score / count) if count > 0 else 0
          print("Predictions to LS:", predictions)
          return [{
            "result": predictions,
            "score": avg_score,
            "model_version": "v8n",  # all predictions will be differentiated by model version
          }]
        except Exception as e:
          print("Exception in predict:", e)
          traceback.print_exc()
          return [{
              "result": [],
              "score": 0,
              "model_version": "v8n",
          }]
        
        
        