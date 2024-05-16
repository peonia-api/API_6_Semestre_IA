import cv2,  time, os
import numpy as np
import requests
from ultralytics import YOLO
 
 
np.random.seed(22)
 
class Deeplearning:
    def __init__(self):
        self.count = 0
        self.line_x = 550
        self.line_height = 1000
        self.person_ids = {}
        self.ponto_esquerdo = (650, 380)
        self.ponto_direito = (280, 200)
        self.fator_reducao = 0.2
        self.novo_ponto_direito = (self.ponto_esquerdo[0] + int((self.ponto_direito[0] - self.ponto_esquerdo[0]) * self.fator_reducao),
                                        self.ponto_esquerdo[1] + int((self.ponto_direito[1] - self.ponto_esquerdo[1]) * self.fator_reducao))
        self.location = (0,0,0,0)
        
 
 
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
 
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
 
    def downloadModel(self, modelUrl):
        fileName = os.path.basename(modelUrl)
        self.modelName = fileName[:fileName.index('.')]
 
        self.cacheDir = './pretrained_models'
 
        os.makedirs(self.cacheDir, exist_ok=True)
        path = os.path.join(self.cacheDir, "checkpoints", fileName)
        if not os.path.exists(path):
            print(f"Downloading model {fileName} ...")
            response = requests.get(modelUrl)
            with open(path, 'wb') as f:
                f.write(response.countent)
            print("Download completed.")
 
    def loadModel(self, model):
        print("Loading Model " + model)

        self.model = YOLO(model)
 
        print("Model " + model + ' loaded success ...')

    def personIO(self, xmin, xmax, ymin, ymax, person_id):
        print('OIIIII')
        print(xmin, xmax, ymin, ymax)
 
        print('Person id: ', person_id)
 
        obj_center_x = (xmin + xmax) / 2
        obj_center_y = (ymin + ymax) / 2
 
        line_margin = 20  
        line_area_start_x = self.line_x - line_margin
        line_area_end_x = self.line_x + line_margin
        line = (self.location[0] + self.location[2]) // 2

        print(obj_center_x)
        print(line)
        print(xmax)
      
     
        if xmax < line and person_id not in self.person_ids.keys():
            self.person_ids[person_id] = 'saida'
        elif xmax < line and person_id in self.person_ids.keys() and self.person_ids[person_id] == 'entrada':
            self.count -= 1 
            del self.person_ids[person_id]
            self.postIO('http://localhost:8082/record', 0, 'Laboratorio')

        if obj_center_x >= line and person_id in self.person_ids.keys() and self.person_ids[person_id] == 'saida':
            self.count += 1
            self.person_ids[person_id] = 'entrada'
            self.postIO('http://localhost:8082/record', 1, 'Laboratorio')
        elif obj_center_x >= line and person_id not in self.person_ids.keys():
            self.person_ids[person_id] = 'entrada'


        # if xmax < self.novo_ponto_direito[0] and person_id not in self.person_ids.keys():
        #     self.person_ids[person_id] = 'saida'
        # elif xmax < self.novo_ponto_direito[0] and person_id in self.person_ids.keys() and self.person_ids[person_id] == 'entrada':
        #     self.count -= 1 
        #     del self.person_ids[person_id]
        #     self.postIO('http://localhost:8082/record', 0, 'Laboratorio')

        # if obj_center_x > line_area_end_x and person_id in self.person_ids.keys() and self.person_ids[person_id] == 'saida':
        #     self.count += 1
        #     self.person_ids[person_id] = 'entrada'
        #     self.postIO('http://localhost:8082/record', 1, 'Laboratorio')
        # elif obj_center_x > line_area_end_x and person_id not in self.person_ids.keys():
        #     self.person_ids[person_id] = 'entrada'

    def createBoundigBox(self, image, threshold=0.5):
        try:           
            trackers = self.model.track(image, persist=True)
            for tracker in trackers:
                boxes = tracker.boxes
                for box in boxes:
                    xmin, ymin, xmax, ymax = box.xyxy[0]
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    w, h = xmax - xmin, ymax - ymin
                    classIndex = int(box.cls[0])
                    classLabelText = self.classesList[classIndex].upper()
                    print(classLabelText)
                    print(self.location)
                    
                    if classLabelText == 'DOOR' and float(box.conf) >= 0.2:
                        self.location = (xmin, ymin, xmin + w, ymin + h)
                    
                    displayText = "{}".format('DOOR')
                    cv2.rectangle(image, (self.location[0], self.location[1]), (self.location[2], self.location[3]), color=(165,100,25), thickness=2)
                    cv2.putText(image, displayText, (self.location[0], self.location[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (165,100,25), 2)
                        

                    if classLabelText == 'PERSON' and float(box.conf) >= threshold:
                        displayText = "{}: {:.2f}%".format(classLabelText, float(box.conf))
                        cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), color=(255, 0, 255), thickness=2)
                        cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)
                        self.personIO(xmin, xmax, w, h, int(box.id.item()))
                    elif self.count == 0:
                        self.person_ids = {}
 
        except Exception as e:
            print(e)
 
        return image
 
 
    def predictImage(self, imagePath, threshold = 0.5):
        image = cv2.imread(imagePath)
 
        bboxImage = self.createBoundigBox(image, threshold)
 
        cv2.imwrite(self.modelName + '.jpeg', bboxImage)
        cv2.imshow("Result ", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
 
    def predictVideo(self, videopath, threshold = 0.5):
        cap = cv2.VideoCapture(videopath)
 
        if(cap.isOpened() == False):
            print("Erro ao abrir ...")
            return
 
        (success, image) = cap.read()
 
        startTime = 0
 
        while success:
            currentTime = time.time()
 
            fps = 1/(currentTime - startTime)
            startTime = currentTime
 
            bboxImage = self.createBoundigBox(image, threshold)
            cv2.line(image, self.ponto_esquerdo, self.novo_ponto_direito, (255, 255, 0), thickness=1)
 
 
            cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
            cv2.putText(bboxImage, "Quantidade: " + str(self.count), (500,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
            cv2.imshow("Result ", bboxImage)
 
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
 
            (success, image) = cap.read()
 
        cv2.destroyAllWindows()
 
 
    def postIO(self,route,occurrence, room):
        try:
            resp = requests.post(route, json={"occurrence": occurrence, "room": room, 'count': self.count})
            print(resp)
        except Exception as e:
            print(f"Erro ao enviar a solicitação HTTP: {e}")