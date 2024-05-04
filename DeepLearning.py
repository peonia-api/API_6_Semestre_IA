import cv2,  time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
import requests
import uuid
from ultralytics import YOLO
 
 
np.random.seed(22)
 
class Deeplearning:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.cont = 0
        self.line_x = 550
        self.line_height = 1000
        self.person_ids = {}
        self.ponto_esquerdo = (650, 380)
        self.ponto_direito = (280, 200)
        self.fator_reducao = 0.2
        self.novo_ponto_direito = (self.ponto_esquerdo[0] + int((self.ponto_direito[0] - self.ponto_esquerdo[0]) * self.fator_reducao),
                                        self.ponto_esquerdo[1] + int((self.ponto_direito[1] - self.ponto_esquerdo[1]) * self.fator_reducao))
        
 
 
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
 
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
 
    def downloadModel(self, modelUrl):
        fileName = os.path.basename(modelUrl)
        self.modelName = fileName[:fileName.index('.')]
 
        self.cacheDir = './pretrained_models'
 
        os.makedirs(self.cacheDir, exist_ok=True)
 
        get_file(fname=fileName, origin=modelUrl, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)
 
    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
 
        print("Model " + self.modelName + ' loaded success ...')
 
 
 
 
 
    def personIO(self, xmin, xmax, ymin, ymax, person_id):
        print('OIIIII')
        print(xmin, xmax, ymin, ymax)
 
        print('Person id: ', person_id)
 
        obj_center_x = (xmin + xmax) / 2
        obj_center_y = (ymin + ymax) / 2
 
        line_margin = 20  
        line_area_start_x = self.line_x - line_margin
        line_area_end_x = self.line_x + line_margin
      
     
        if xmax < self.novo_ponto_direito[0] and person_id not in self.person_ids.keys():
            self.person_ids[person_id] = 'saida'
        elif xmax < self.novo_ponto_direito[0] and person_id in self.person_ids.keys() and self.person_ids[person_id] == 'entrada':
            self.cont -= 1 
            del self.person_ids[person_id]
            self.postIO('http://localhost:8080/record', 0, 'Laboratorio')

        if obj_center_x > line_area_end_x and person_id in self.person_ids.keys() and self.person_ids[person_id] == 'saida':
            self.cont += 1
            self.person_ids[person_id] = 'entrada'
            self.postIO('http://localhost:8080/record', 1, 'Laboratorio')
        elif obj_center_x > line_area_end_x and person_id not in self.person_ids.keys():
            self.person_ids[person_id] = 'entrada'
            
 
 
    def createBoundigBox(self, image, threshold=0.5):
        try:           
            results = self.model.track(image, persist=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    classIndex = int(box.cls[0])
                    classLabelText = self.classesList[classIndex].upper()
                    print(classLabelText)
                    acuracia = float(box.conf) 
                    if classLabelText == 'PERSON' and acuracia >= threshold:
                        displayText = "{}: {:.2f}%".format(classLabelText, acuracia)
                        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color=(255, 0, 255), thickness=2)
                        cv2.putText(image, displayText, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)
                        print( x1, y1, x2, y2)
                        self.personIO(x1, x2, w, h, int(box.id.item()))
                    elif self.cont == 0:
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
            cv2.putText(bboxImage, "Quantidade: " + str(self.cont), (500,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
            cv2.imshow("Result ", bboxImage)
 
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
 
            (success, image) = cap.read()
 
        cv2.destroyAllWindows()
 
 
    def postIO(self,route,occurrence, room):
        try:
            resp = requests.post(route, json={"occurrence": occurrence, "room": room, 'cont': self.cont})
            print(resp)
        except Exception as e:
            print(f"Erro ao enviar a solicitação HTTP: {e}")