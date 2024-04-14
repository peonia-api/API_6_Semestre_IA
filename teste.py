import cv2
import numpy as np
import tensorflow as tf

# Função para verificar se uma pessoa está dentro da ROI
def pessoa_dentro_roi(frame, roi):
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]

    # Convertendo o frame ROI para o espaço de cores HSV
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Definindo os intervalos de cores para detectar a cor amarela
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Criando uma máscara para isolar os pixels amarelos na ROI
    mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Encontrando contornos na máscara
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Se houver contornos (presença de cor amarela), consideramos que uma pessoa está presente na ROI
    if len(contornos) > 0:
        return True
    else:
        return False


# Função para detecção de objetos
def detect_objects(frame):
    # Redimensionar o frame para o tamanho esperado pelo modelo
    resized_frame = cv2.resize(frame, (64, 64))

    # Pré-processar o frame (normalização)
    processed_frame = resized_frame / 255.0

    # Expandir as dimensões do frame para corresponder à entrada esperada pelo modelo
    input_tensor = np.expand_dims(processed_frame, axis=0)

    # Realizar a detecção de objetos
    predictions = model.predict(input_tensor)

    # Iterar sobre as previsões e extrair as caixas delimitadoras e as classes
    for prediction in predictions:
        bbox = prediction[:4]  # As primeiras quatro coordenadas representam a caixa delimitadora (x, y, w, h)

        # Ajustar as coordenadas da caixa delimitadora ao tamanho original do frame
        bbox[0] *= resized_frame.shape[1]  # x
        bbox[1] *= resized_frame.shape[0]  # y
        bbox[2] *= resized_frame.shape[1]  # w
        bbox[3] *= resized_frame.shape[0]  # h

        # Converter as coordenadas para inteiros
        bbox = bbox.astype(int)

        # Extrair a classe prevista
        classe_prevista = np.argmax(prediction[4:])

        # Desenhar a caixa delimitadora no frame
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'Classe: {classe_prevista}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Carregar o modelo treinado
model = tf.keras.models.load_model('modelo.h5')

# Acessar a câmera
video = cv2.VideoCapture(0)

# Definir a posição anterior da pessoa na ROI para cada lado (entrada e saída)
posicao_anterior_entrada = None
posicao_anterior_saida = None

# Definir contadores
contador_entrada = 0
contador_saida = 0

# Definir uma margem de erro para considerar que a pessoa cruzou a linha de divisão
margem_de_erro = 50

while True:
    # Capturar frame por frame
    ret, frame = video.read()
    if not ret:
        break

    # Adicionar uma linha na frente da tela para representar a divisão entre a área de entrada e saída
    altura, largura, _ = frame.shape
    linha_divisoria = largura // 2
    cv2.line(frame, (linha_divisoria, 0), (linha_divisoria, altura), (255, 255, 255), 2)

    # Definir regiões de interesse (ROIs) para entrada e saída
    margem_entrada = 50
    margem_saida = 50
    roi_entrada = [linha_divisoria - margem_entrada, 0, margem_entrada, altura]
    roi_saida = [linha_divisoria, 0, margem_saida, altura]

    cv2.line(frame, (linha_divisoria - margem_entrada, 0), (linha_divisoria - margem_entrada, altura), (43, 234, 255), 2)
    cv2.line(frame, (linha_divisoria + margem_saida, 0), (linha_divisoria + margem_saida, altura), (0, 0, 255), 2)

    # Pré-processamento do frame
    frame_redimensionado = cv2.resize(frame, (64, 64)) / 255.0
    frame_redimensionado = np.expand_dims(frame_redimensionado, axis=0)

    # Detecção de objetos
    processed_frame = detect_objects(frame)

    # Mostrar o frame com as detecções
    cv2.imshow('Frame', processed_frame)

    # Verificar se uma pessoa entrou na ROI de entrada
    if pessoa_dentro_roi(frame, roi_entrada):
        if posicao_anterior_entrada is None or posicao_anterior_entrada == "outside":
            contador_entrada += 1
            posicao_anterior_entrada = "inside"
            print("Entrada")
    else:
        if posicao_anterior_entrada == "inside":
            posicao_anterior_entrada = "outside"

    # Verificar se uma pessoa saiu da ROI de saída
    if pessoa_dentro_roi(frame, roi_saida):
        if posicao_anterior_saida is None or posicao_anterior_saida == "outside":
            contador_saida += 1
            posicao_anterior_saida = "inside"
            print("Saída")
    else:
        if posicao_anterior_saida == "inside":
            posicao_anterior_saida = "outside"

    # Mostrar os contadores
    print(f"Entradas: {contador_entrada}, Saídas: {contador_saida}")

    # Verificar se o usuário pressionou 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o objeto de captura e fechar a janela
video.release()
cv2.destroyAllWindows()