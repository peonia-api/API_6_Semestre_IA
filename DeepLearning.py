import cv2,  time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
import requests
import uuid
from deepsort.tracker import DeepSortTracker
from deepsort.detection import Detection


np.random.seed(22)

class Deeplearning:
    def __init__(self):
        self.cont = 0
        self.line_x = 550  
        self.line_height = 1000  
        self.person_ids = set()
        self.ponto_esquerdo = (650, 380)  
        self.ponto_direito = (280, 200)
        self.fator_reducao = 0.2
        self.novo_ponto_direito = (self.ponto_esquerdo[0] + int((self.ponto_direito[0] - self.ponto_esquerdo[0]) * self.fator_reducao),
                                        self.ponto_esquerdo[1] + int((self.ponto_direito[1] - self.ponto_esquerdo[1]) * self.fator_reducao))
        self.person_info = {}
        self.rastreador = DeepSortTracker(max_age=30, metric_name='euclidean',max_iou_distance=0.7,n_init=3, max_dist=0.2, nn_budget=100)


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
  
        # Define uma área ao redor da linha onde a pessoa é considerada para contagem
        line_margin = 20  # Margem em pixels
        line_area_start_x = self.line_x - line_margin
        line_area_end_x = self.line_x + line_margin

        print(line_area_start_x)
        print(line_area_end_x)

        print(obj_center_x)
        if obj_center_x > self.novo_ponto_direito[0]:
            self.cont += 1


        # print(f'Verificação {line_area_start_x <= obj_center_x <= line_area_end_x }')
        # if line_area_start_x <= obj_center_x <= line_area_end_x:
        #     print("A pessoa está na linha.")
        #     if obj_center_x > self.line_x:
        #         print("A pessoa está indo para a direita.")
        #         self.cont += 1
        #     elif obj_center_x < self.line_x and line_area_start_x <= obj_center_x:
        #         print("A pessoa está indo para a esquerda.")
        #         self.cont -= 1
        # if line_area_start_x <= obj_center_x <= line_area_end_x:
        #     print(xmax < self.line_x)
        #     # Verifica se a pessoa está indo da direita para a esquerda (entrada)
        #     if xmax < self.line_x:
        #         print("Pessoa saiu pela porta!")
        #         self.cont -= 1  # Decrementa o contador de saída
               
        #     if xmax > self.line_x:
        #         print("Pessoa entrou pela porta!")
        #         self.cont += 1  # Incrementa o contador de entrada
                
            # Verifica se a pessoa está indo da esquerda para a direita (saída)
            
        # print(self.person_ids)

    def is_same_person(self, info, xmin, xmax, ymin, ymax):
        # Coordenadas da caixa delimitadora atual
        box1 = [xmin, ymin, xmax, ymax]

        # Coordenadas da caixa delimitadora previamente detectada
        box2 = [info['xmin'], info['ymin'], info['xmax'], info['ymax']]

        # Calcular a área da interseção
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return False  # Não há sobreposição

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calcular a área de ambas as caixas delimitadoras
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calcular a relação de sobreposição
        overlap = intersection_area / float(min(area_box1, area_box2))

        # Calcular a distância entre os centros das caixas delimitadoras
        center_distance = ((box1[0] + box1[2]) / 2 - (box2[0] + box2[2]) / 2) ** 2 + ((box1[1] + box1[3]) / 2 - (box2[1] + box2[3]) / 2) ** 2
        center_distance = center_distance ** 0.5

        # Se a relação de sobreposição e a distância entre os centros forem menores que os limiares, consideramos que é a mesma pessoa
        overlap_threshold = 0.5  # Ajuste conforme necessário
        center_distance_threshold = 100  # Ajuste conforme necessário

        return overlap > overlap_threshold and center_distance < center_distance_threshold


    def createBoundigBox(self, image, threshold=0.5):
        try:
            inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
            inputTensor = inputTensor[tf.newaxis, ...]

            detections = self.model(inputTensor)

            bboxs = detections['detection_boxes'][0]
            classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
            classScores = detections['detection_scores'][0].numpy()

            

            imH, imW, imC = image.shape

            bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
                                                    iou_threshold=threshold, score_threshold=threshold)

            if len(bboxIdx) != 0:
                for i in bboxIdx:
                    # bbox = tuple(bboxs[i].tolist())
                    bbox = bboxs[i]
                    classConfidence = round(100*classScores[i])
                    classIndex = classIndexes[i]

                    classLabelText = self.classesList[classIndex].upper()


                    classColor = self.colorList[classIndex]

                    

                    ymin, xmin, ymax, xmax = bbox

                    xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                    print(bbox)
                    if classLabelText == 'PERSON':
                        # assignments, scores = self.rastreador.update_tracks(bboxs, classScores)  # Realize a associação de detecções
                       

# Create a Detection object
                        
                        detection = Detection(tlwh=[xmin, ymin, xmax, ymax], confidence=0.2, feature=None)
                        xyah = detection.to_xyah()
                        print(xyah)
                        online_targets = self.rastreador.update(xyah)
                        print(online_targets)
                        #track_id = assignments[i]

                        #print(track_id)

                        #self.personIO(xmin, xmax, ymin, ymax, track_id)
                        # displayText = "{}: {}%".format(classLabelText, classConfidence)
                        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                        # cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                        # lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))

                        # cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                        # cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)

                        # cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                        # cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)

                        # cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                        # cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)

                        # cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                        # cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)
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

    
    def postIO(route,occurrence, room):
        try:
            resp = requests.post(route, json={"occurrence": occurrence, "room": room})
            print(resp)
        except Exception as e:
            print(f"Erro ao enviar a solicitação HTTP: {e}")