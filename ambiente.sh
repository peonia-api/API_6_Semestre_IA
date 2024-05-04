python3 --version > /dev/null 2>&1

if [ $? -gt 0 ]
then
	echo -e "Instalando o python 3.10!\nIsso vai demorar alguns minutos!"
	sleep 3
	sudo apt install python3.10 python3-pip -y 
	if [ $? -eq 0 ]
	then
		echo "Python instalado com sucesso!"
	else
		echo "Não foi possivel instalar o python!"
		exit 1
	fi
else
	echo "Python já esta instalado!"
fi

pip3 install numpy opencv_contrib_python opencv_python Requests ultralytics
if [ $? -eq 0 ]
then
    clear
    echo "Instalado com sucesso!"
    echo "Deseja rodar o programa?\n[ y ] Yes\n[ n ] No"
    read -p "Digite aqui: " OPECAO

    case $OPECAO in 
        y)
            echo "Rodando o programa ..."
            python3 ./ia-camera.py
            ;;
        n)
            exit 0
            ;;
        *)
            echo "Opção invalida!"
            ;;
        esac
else
    echo "Não foi possivel instalar!"
    exit 1
fi
