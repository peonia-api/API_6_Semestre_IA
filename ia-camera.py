# from DeepLearning import *

# #modelo = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

# modelo = 'best.pt'

# classFile = "names.names"
# imagePath = "/home/mateus/Imagens/ps.webp"
# videoPath = "entrada02.mp4"

# threshold = 0.3


# detector = Deeplearning()
# detector.readClasses(classFile)

# #detector.downloadModel(modelo)
# detector.loadModel(modelo)
# #detector.predictImage(imagePath, threshold)
# detector.predictVideo(videoPath, 'Laboratorio' ,threshold)

import os
import subprocess
import multiprocessing
import shutil



data = [
    {
        "url": 'entrada02.mp4',
        "zone": 'Laboratorio'
    },
    {
        "url": 'entrada03.mp4',
        "zone": 'Cozinha'
    }
]

def arquivo(nome, url, folder_path):

    # Caminho para o novo arquivo
    file_path = os.path.join(folder_path, f'{nome}.py')

    script = f"""
from DeepLearning import *

modelo = 'best.pt'

classFile = "names.names"
videoPath = "{url}"

threshold = 0.3

detector = Deeplearning()
detector.readClasses(classFile)
detector.loadModel(modelo)
detector.predictVideo(videoPath, "{nome}" ,threshold)
    """

    # Cria e escreve em um novo arquivo
    with open(file_path, 'w') as file:
        file.write(script)








def run_script(script_path):
    """Função para executar um script Python."""
    subprocess.run(['python', script_path])

def main():
    # Caminho para a nova pasta
    folder_path = 'readzones'

    # Cria a pasta (e subpastas) se não existirem
    os.makedirs(folder_path, exist_ok=True)
    shutil.copy('DeepLearning.py', 'readzones')
    shutil.copy('names.names', 'readzones')
    shutil.copy('entrada03.mp4', 'readzones')
    shutil.copy('entrada04.mp4', 'readzones')
    
    for item in data:
        arquivo(item['zone'], item['url'], folder_path)
    
    scripts = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.py')]

    processes = []
    for script in scripts:
        process = multiprocessing.Process(target=run_script, args=(script,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()
