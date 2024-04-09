import tensorflow as tf
import numpy as np
import cv2
import uuid

# Inicializar o rastreador Optical Flow
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Carregar o modelo pré-treinado
model = tf.keras.models.load_model('modelo.h5')

# Verificar o tamanho esperado da entrada do modelo
input_shape = model.input_shape[1:3]  # Ignorando o primeiro eixo (tamanho do lote)
print("Tamanho esperado da entrada do modelo:", input_shape)

# Inicializar a câmera
cap = cv2.VideoCapture(0)
object_ids_dict = {}

threshold = 0.5
margem_de_erro = 50

# Capturar o primeiro frame do vídeo
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

while True:
    # Capturar frame da câmera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Redimensionar o frame para o tamanho esperado pelo modelo
    frame_redimensionado = cv2.resize(frame, input_shape)

    altura, largura, _ = frame.shape
    linha_divisoria = largura // 2
    cv2.line(frame, (linha_divisoria, 0), (linha_divisoria, altura), (255, 255, 255), 2)

    # Definir regiões de interesse (ROIs) para entrada e saída
    margem_entrada = 100
    margem_saida = 100
    roi_entrada = [linha_divisoria - margem_entrada, 0, margem_entrada, altura]
    roi_saida = [linha_divisoria, 0, margem_saida, altura]

    cv2.line(frame, (linha_divisoria - margem_entrada, 0), (linha_divisoria - margem_entrada, altura), (43, 234, 255), 2)
    cv2.line(frame, (linha_divisoria + margem_saida, 0), (linha_divisoria + margem_saida, altura), (0, 0, 255), 2)

    
    # Preparar o frame de entrada
    image = cv2.cvtColor(frame_redimensionado, cv2.COLOR_BGR2RGB)
    image_input = np.expand_dims(image, axis=0)
    
    # Executar a detecção de objetos
    detections = model.predict(image_input)
    print(detections)
    
    # Lista para armazenar os identificadores únicos dos objetos detectados
    object_ids = []
    
    # Exemplo de visualização dos resultados
    # Exemplo de visualização dos resultados
    for detection in detections:
        score = detection[0]  # A pontuação de confiança está no primeiro elemento da lista interna
        
        print(frame_redimensionado)
        # Verificar se a detecção possui uma pontuação de confiança suficiente
        if score > threshold:
            # Gerar um identificador único para o objeto
            object_id = str(uuid.uuid4())
            
            # Adicionar o identificador único à lista
            object_ids.append(object_id)
            
            # Obter as coordenadas da caixa delimitadora do objeto detectado
            bbox = (20,20,20,20)  # Supondo que as coordenadas da caixa delimitadora sigam após a pontuação de confiança
            
            # Certifique-se de que a caixa delimitadora não está vazia
            if bbox:
                # Calcular o centro da caixa delimitadora
                center_x = int(bbox[0] + bbox[2] / 2)
                center_y = int(bbox[1] + bbox[3] / 2)
                
                # Gerar uma chave única com base nas coordenadas do centro da caixa delimitadora
                key = (center_x, center_y)
                
                # Verificar se a chave já existe no dicionário
                if key in object_ids_dict:
                    # Se a chave existe, associar o mesmo ID ao objeto
                    object_id = object_ids_dict[key]
                else:
                    # Se a chave não existe, gerar um novo ID para o objeto
                    object_id = str(uuid.uuid4())
                    object_ids_dict[key] = object_id

                print(center_x)
                
                if center_x < linha_divisoria - margem_entrada and center_x > linha_divisoria - margem_entrada - margem_de_erro:
                    print("Saiu")
                   # cv2.imwrite(f'saiu_{object_id}.jpg', frame)
                elif center_x > linha_divisoria + margem_saida and center_x < linha_divisoria + margem_saida + margem_de_erro:
                    print("Entrou")
                    #cv2.imwrite(f'entrou_{object_id}.jpg', frame)

                # Atualizar os pontos de interesse para o próximo frame
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                
                # Selecionar os pontos que foram rastreados com sucesso
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                # Atualizar os pontos de interesse para o próximo frame
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                
                # Desenhar a caixa delimitadora do objeto
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f'Object ID: {object_id}, Score: {score}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibir o frame com as caixas delimitadoras
    cv2.imshow('Object Detection', frame)
    
    # Parar o loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
