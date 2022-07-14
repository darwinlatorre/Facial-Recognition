#OpenCV module
#Libreria de software de machine learnig y vision por comptadora 
import cv2
#Modulo para leer directorios y rutas de archivos
import os
#OpenCV trabaja con arreglos de numpy
#NumPy es una libreria para la computacion cientifica, proporciona diversos objetos y rutinas
import numpy
#Obtener el nombre de la persona que estamos capturando
#Leer por linea de comandos el nombre del sujeto a ser capturado
import sys
nombre = sys.argv[1]

#Directorio donde se encuentra la carpeta con el nombre de la persona
dir_faces = 'att_faces/orl_faces'
path = os.path.join(dir_faces, nombre)

#Tamaño para reducir a miniaturas las fotografias
size_miniatura = 4

#Si no hay una carpeta con el nombre ingresado entonces se crea
if not os.path.isdir(path):
	os.mkdir(path)

#cargamos la plantilla e inicializamos la webcam:
#Pretrain model for face detection
face_cascade_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Pretrain model for eye detection
eye_cascade_model = cv2.CascadeClassifier('haarcascade_eye.xml')
#Captura de video con OPENCV por medio de un host o camara local(0)
video_cap = cv2.VideoCapture('http://192.168.20.22:8080/video')

#Designa los valores del tamaño de la imagen
img_width, img_height = 112, 92
#Ciclo para tomar fotografias
count = 0
# while count < 100:
while(True):

	#leemos un frame y lo guardamos, y se guarda en 'img'
	ret, frame_img = video_cap.read()
	#flip horizontal a la imagen @prm(img, 1), vertical @prm(img, 0)
	frame_img = cv2.flip(frame_img, 1, 0)

	#convertimos la imagen a blanco y negro
	gray_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
	#redimensionar la imagen, shape[1] medidas en x, shape[0] medidas en y
	mini_img = cv2.resize(gray_img, (int(gray_img.shape[1] / size_miniatura), int(gray_img.shape[0] / size_miniatura)))

		#buscamos las coordenadas de los rostros (si los hay) y
		#guardamos su posicion
	#Detecta los direfentes tamaños de la imagen de entrada, y los objetos identificados del modelo so devultos en una lista de rectangulos
	#Retorna 4 valores, 1.CordX, 2.CordY, Width, Height
	faces_height = face_cascade_model.detectMultiScale(mini_img)

	#Devuelve y acomoda por orden el valor de la altura de los rectangulos
	faces_height = sorted(faces_height, key=lambda x: x[3])

	if faces_height:
		face_i = faces_height[0]
		#Adjunta el tamaño en cordenadas del el rectangulo en la imagen
		(x, y, w, h) = [v * size_miniatura for v in face_i]
		#Recortamos imagen en un rectangulo
		face = gray_img[y:y + h, x:x + w]
		#rejustamos el tamaño de la imagen
		face_resize = cv2.resize(face, (img_width, img_height))

		#Dibujamos un rectangulo en las coordenadas del rostro
		cv2.rectangle(frame_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

		#Recortamos imagen en un rectangulo
		roi_gray = gray_img[y:y + h, x:x + w]
		roi_color = frame_img[y:y + h, x:x + w]
		#Detecta los direfentes tamaños de la imagen de entrada, y los objetos identificados del modelo so devultos en una lista de rectangulos
		#Retorna 4 valores, 1.CordX, 2.CordY, Width, Height
		eyes = eye_cascade_model.detectMultiScale(roi_gray)
		#Recojemos las cordenadas de los ojos
		for (ex, ey, ew, eh) in eyes:
			#Dibujamos un rectangulo en las coordenadas del rostro
			cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


		#Ponemos el nombre en el rectagulo
		cv2.putText(frame_img, nombre, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))        

		#El nombre de cada foto es el numero del ciclo
		#Obtenemos el nombre de la foto
		#Despues de la ultima sumamos 1 para continuar con los demas nombres
		pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
				if n[0]!='.' ]+[0])[-1] + 1

		#Metemos la foto en el directorio
		cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

		#Contador del ciclo
		count += 1

	#Mostramos la imagen
	cv2.imshow('OpenCV Entrenamiento de '+nombre, frame_img)

	#Si se presiona la tecla ESC se cierra el programa
	key = cv2.waitKey(10)
	if key == 27:
		cv2.destroyAllWindows()
		break
