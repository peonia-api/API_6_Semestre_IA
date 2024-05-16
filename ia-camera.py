from DeepLearning import *

#modelo = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz'

#modelo = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"

#modelo = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"

modelo = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

classFile = "names.names"
imagePath = "/home/mateus/Imagens/ps.webp"
videoPath = "person.mp4"

threshold = 0.4


detector = Deeplearning()
detector.readClasses(classFile)

#detector.downloadModel(modelo)
detector.loadModel('best.pt')
#detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath, threshold)