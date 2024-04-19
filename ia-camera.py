from DeepLearning import *

#modelo = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz'

modelo = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"

classFile = "coco.names"
imagePath = "/home/mateus/Imagens/ps.webp"
threshold = 0.5


detector = Deeplearning()
detector.readClasses(classFile)

detector.downloadModel(modelo)
detector.loadModel()
detector.predictImage(imagePath, threshold)