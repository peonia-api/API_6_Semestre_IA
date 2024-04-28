import os
import io
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

def create_tf_example(group, image_path):
    # Carregar a imagem para obter suas dimensões
    image = Image.open(os.path.join(image_path, group['filename']))
    width, height = image.size

    # Codificar a imagem em formato JPEG
    with tf.io.gfile.GFile(os.path.join(image_path, group['filename']), 'rb') as fid:
        encoded_jpg = fid.read()

    filename = group['filename'].encode('utf8')
    image_format = b'jpg'
    xmins = [group['xmin'] / width]
    xmaxs = [group['xmax'] / width]
    ymins = [group['ymin'] / height]
    ymaxs = [group['ymax'] / height]
    classes_text = [group['class'].encode('utf8')]
    classes = [class_text_to_int(group['class'])]

    # Criar exemplo do TensorFlow
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def class_text_to_int(row_label):
    class_dict = {"person": 1, "door": 2}
    return class_dict.get(row_label, -1)

def xml_to_data(path):
    data = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        for member in root.findall('object'):
            value = {
                'filename': filename,
                'width': int(root.find('size')[0].text),
                'height': int(root.find('size')[1].text),
                'class': member[0].text,
                'xmin': int(member.find('bndbox').find('xmin').text),
                'ymin': int(member.find('bndbox').find('ymin').text),
                'xmax': int(member.find('bndbox').find('xmax').text),
                'ymax': int(member.find('bndbox').find('ymax').text)
            }
            data.append(value)
    return data

def main():
    for folder in ['train', 'val']:
        xml_path = os.path.join(os.getcwd(), 'dataset/annotations/{}'.format(folder))
        image_path = os.path.join(os.getcwd(), 'dataset/imagens/{}'.format(folder))
        print(xml_path)
        if not os.path.exists('dataset/annotations'):
            os.makedirs('dataset/annotations')
        
        xml_data = xml_to_data(xml_path)
        print("XML files:", [item['filename'] for item in xml_data])

        examples = [create_tf_example(item, image_path) for item in xml_data]
        output_path = os.path.join(os.getcwd(), 'dataset/annotations/{}_record'.format(folder))
        with tf.io.TFRecordWriter(output_path) as writer:
            for example in examples:
                if example:
                    writer.write(example.SerializeToString())
        print('Successfully created the TFRecord file for {}.'.format(folder))

if __name__ == '__main__':
    main()
