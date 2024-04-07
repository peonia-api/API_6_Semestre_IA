import cv2
import numpy as np
import tensorflow as tf

# Função para verificar se uma pessoa está dentro da ROI
def pessoa_dentro_roi(frame, roi):
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]

    # Convertendo o frame ROI para o espaço de cores HSV
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Definindo os intervalos de cores para detectar a cor vermelha
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Criando uma máscara para isolar os pixels vermelhos na ROI
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Encontrando contornos na máscara
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Se houver contornos (presença de cor vermelha), consideramos que uma pessoa está presente na ROI
    if len(contornos) > 0:
        return True
    else:
        return False

# Carregar o modelo treinado
modelo = tf.keras.models.load_model('modelo')

# Acessar a câmera
video = cv2.VideoCapture(0)

# Definir a posição anterior da pessoa na ROI para cada lado (entrada e saída)
posicao_anterior_entrada = None
posicao_anterior_saida = None

# Definir contadores
contador_entrada = 0
contador_saida = 0

# Definir uma margem de erro para considerar que a pessoa cruzou a linha de divisão
margem_de_erro = 20

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
    margem_entrada = 30
    margem_saida = 30
    roi_entrada = [linha_divisoria - margem_entrada, 0, margem_entrada, altura]
    roi_saida = [linha_divisoria, 0, margem_saida, altura]

    # Pré-processamento do frame
    frame_redimensionado = cv2.resize(frame, (64,64)) / 255.0
    frame_redimensionado = np.expand_dims(frame_redimensionado, axis=0)

    # Detecção de pessoas
    predict = modelo.predict(frame_redimensionado)
    classe_prevista = np.argmax(predict)
    print(classe_prevista)
    # Verificar se uma pessoa foi detectada
    if classe_prevista == 1:
        # Verificar a presença de uma pessoa na ROI antes de tentar verificar a direção do movimento
        if pessoa_dentro_roi(frame, roi_entrada) or pessoa_dentro_roi(frame, roi_saida):
            if posicao_anterior_entrada is not None and pessoa_dentro_roi(frame, roi_entrada):
                posicao_atual_entrada = (linha_divisoria, frame.shape[0] // 2)
                print(posicao_atual_entrada[0] < posicao_anterior_entrada[0] + margem_de_erro)

                print("Posicao anterior entrada:", posicao_anterior_entrada)
                print("Posicao atual entrada:", posicao_atual_entrada)
                if posicao_atual_entrada[0] < posicao_anterior_entrada[0] + margem_de_erro:
                    contador_entrada += 1
                    print("Pessoa entrando")
            elif posicao_anterior_saida is not None and pessoa_dentro_roi(frame, roi_saida):
                posicao_atual_saida = (linha_divisoria, frame.shape[0] // 2)
                if posicao_atual_saida[0] > posicao_anterior_saida[0] - margem_de_erro:
                    contador_saida += 1
                    print("Pessoa saindo")
    else:
        # Resetar a posição anterior se nenhuma pessoa for detectada
        posicao_anterior_entrada = None
        posicao_anterior_saida = None

    # Atualizar a posição anterior apenas se uma pessoa foi detectada
    if pessoa_dentro_roi(frame, roi_entrada) or pessoa_dentro_roi(frame, roi_saida):
        posicao_anterior_entrada = (linha_divisoria, frame.shape[0] // 2)
        posicao_anterior_saida = (linha_divisoria, frame.shape[0] // 2)

    # Mostrar os contadores
    print(f"Entradas: {contador_entrada}, Saídas: {contador_saida}")

    # Mostrar o frame com as detecções
    cv2.imshow('Frame', frame)

    # Verificar se o usuário pressionou 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o objeto de captura e fechar a janela
video.release()
cv2.destroyAllWindows()
