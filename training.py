import os
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import tensorflow as tf

cache_dir = './pretrained_models'

# Caminho para o TensorFlow Object Detection API
PATH_TO_OBJECT_DETECTION_API = os.path.join(cache_dir, 'checkpoints')

# Caminho para o diretório de trabalho
PATH_TO_WORKSPACE = '/home/mateus/Documentos/API/API_6_Semestre_IA/'

# Caminho para o diretório de anotações do dataset
PATH_TO_ANNOTATIONS = os.path.join(PATH_TO_WORKSPACE, 'dataset', 'annotations')

# Caminho para o diretório de imagens do dataset
PATH_TO_IMAGES = os.path.join(PATH_TO_WORKSPACE, 'dataset', 'imagens')

# Caminho para o arquivo de labels
PATH_TO_LABELS = os.path.join(PATH_TO_WORKSPACE, 'dataset', 'annotations', 'label_map.pbtxt')

# Caminho para o arquivo de configuração do modelo
PATH_TO_CONFIG = os.path.join(PATH_TO_OBJECT_DETECTION_API, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8' ,'pipeline.config')

# Número de classes no seu dataset (incluindo a classe de fundo)
NUM_CLASSES = 1

# Atualize o arquivo de configuração do modelo para refletir o número de classes
def update_config(num_classes):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(PATH_TO_CONFIG, "r") as f:
        proto_str = f.read()
        # Remova o campo 'fine_tune_checkpoint_version' do texto do arquivo de configuração
        proto_str = proto_str.replace('fine_tune_checkpoint_version: V2', '')

        # Faça a fusão das configurações
        text_format.Merge(proto_str, pipeline_config)
    
    pipeline_config.model.ssd.num_classes = num_classes
    pipeline_config.model.ssd.freeze_batchnorm = True  # Adicione isso se desejar congelar a normalização em lote
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PATH_TO_OBJECT_DETECTION_API, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = PATH_TO_LABELS
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(PATH_TO_ANNOTATIONS, 'train_record')]
    pipeline_config.eval_input_reader[0].label_map_path = PATH_TO_LABELS
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(PATH_TO_ANNOTATIONS, 'val_record')]
    
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(PATH_TO_CONFIG, "wb") as f:
        f.write(config_text)


# Treinamento do modelo
def train_model():
    #os.chdir(PATH_TO_OBJECT_DETECTION_API)
    os.system(f'python3 model_main_tf2.py --model_dir={PATH_TO_WORKSPACE}/pretrained_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 --pipeline_config_path={PATH_TO_CONFIG}')

# Função principal
def main():
    update_config(NUM_CLASSES)
    train_model()

if __name__ == "__main__":
    main()
