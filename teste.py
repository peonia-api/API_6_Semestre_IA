import tensorflow as tf
import numpy as np
import cv2
import requests

# Carregar o modelo pré-treinado
model = tf.keras.models.load_model('modelo')

# Definir o tamanho esperado da entrada do modelo
input_shape = model.input_shape[1:3]
print("Tamanho esperado da entrada do modelo:", input_shape)

# Inicializar a câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a câmera!")
    exit()

threshold = 0.5  # Limiar de confiança
previous_value = 0  # Armazenar o valor anterior para comparação

print("Iniciando loop principal...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar imagem da câmera.")
        break

    # Desenhar as linhas de divisão no frame original
    height, width = frame.shape[:2]
    cv2.line(frame, (width//3, 0), (width//3, height), (255, 0, 0), 2)  # Linha da zona de entrada
    cv2.line(frame, (2*width//3, 0), (2*width//3, height), (0, 255, 0), 2)  # Linha da zona de saída

    # Redimensionar o frame para o tamanho esperado pelo modelo para processamento
    frame_resized = cv2.resize(frame, input_shape)
    image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    image_input = np.expand_dims(image, axis=0)

    # Executar a detecção de objetos
    detections = model.predict(image_input)
    print("Detections:", detections)  # Imprimir detecções para verificar

    # Processar cada detecção
    for detection in detections:
        print("oi")
        confidence, value = detection
        if confidence > threshold:  # Verificar se a pontuação de confiança é alta o suficiente
            print("Object detected with high confidence:", confidence)
            print(value > previous_value)
            print(value < previous_value)
            if value > previous_value:
                print("Entrada detectada")
                try:
                    resp = requests.post("http://localhost:8080/record", json={"occurrence": "1", "room": "Laboratorio"})
                    print(resp)
                except Exception as e:
                    print(f"Não foi {e}")
            elif value < previous_value:
                print("Saída detectada")
                try:
                    resp = requests.post("http://localhost:8080/record", json={"occurrence": "0", "room": "Laboratorio"})
                    print(resp)
                except Exception as e:
                    print(f"Não foi {e}")
            previous_value = value  # Atualizar o valor anterior para comparações futuras

    # Exibir o frame capturado para verificar se a câmera está funcionando
    cv2.imshow('Camera Output', frame)

    # Verificar se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar todas as janelas abertas
cap.release()
cv2.destroyAllWindows()
