#OpenCV module
#Libreria de software de machine learnig y vision por comptadora 
import cv2
#Modulo para leer directorios y rutas de archivos
import os
#OpenCV trabaja con arreglos de numpy
#NumPy es una libreria para la computacion cientifica, proporciona diversos objetos y rutinas
import numpy
#Se importa la lista de personas con acceso al laboratorio
from listaPermitidos import flabianos
flabs=flabianos()

# Parte 1: Creando el entrenamiento del modelo
#Directorio donde se encuentran las carpetas con las caras de entrenamiento
dir_faces = 'att_faces/orl_faces'
#Tamaño para reducir a miniaturas las fotografias
size_miniatura = 4

# Crear una lista de imagenes y una lista de nombres correspondientes
(lista_images, lables_id, names_subdir, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(dir_faces):
	for subdir in dirs:
		names_subdir[id] = subdir
		subjectpath = os.path.join(dir_faces, subdir)
		for filename in os.listdir(subjectpath):
			path = subjectpath + '/' + filename
			lable = id
			lista_images.append(cv2.imread(path, 0))
			lables_id.append(int(lable))
		id += 1

#Designa los valores del tamaño de la imagen
(im_width, im_height) = (112, 92)

#TODO: POR ENTEDER 
# Crear una matriz Numpy de las dos listas anteriores
(lista_images, lables_id) = [numpy.array(lis) for lis in [lista_images, lables_id]]
# OpenCV entrena un modelo a partir de las imagenes
# LBP(Local Binary Patterns)
# OpenCV transforms LBP images to histograms to store spatial information
model = cv2.face.LBPHFaceRecognizer_create()
model.train(lista_images, lables_id)


# Parte 2: Utilizar el modelo entrenado en funcionamiento con la camara
# Pretrain model for face detection
face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')
# Pretrain model for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Captura de video con OPENCV por medio de un host o camara local(0)
cap = cv2.VideoCapture('http://192.168.20.22:8080/video')

# Caffe (Convolutional Architecture for Fast Feature Embedding) is a 
# deep learning framework that allows users to create image classification 
# and image segmentation models
age_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_age.prototxt', 
		'data/age_net.caffemodel')

gender_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_gender.prototxt', 
		'data/gender_net.caffemodel')


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

while True:
	#leemos un frame y lo guardamos, y se guarda en 'img'
	rval, frame = cap.read()
	#flip horizontal a la imagen @prm(img, 1), vertical @prm(img, 0)
	frame=cv2.flip(frame,1,0)

	#convertimos la imagen a blanco y negro    
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#redimensionar la imagen, shape[1] medidas en x, shape[0] medidas en y
	mini = cv2.resize(gray, (int(gray.shape[1] / size_miniatura), int(gray.shape[0] / size_miniatura)))

	"""buscamos las coordenadas de los rostros (si los hay) y
		guardamos su posicion"""
	#TODO: POR ENTEDER 
	faces = face_cascade.detectMultiScale(mini)

	#TODO: POR ENTEDER 
	for i in range(len(faces)):
		face_i = faces[i]
		#Adjunta el tamaño en cordenadas del el rectangulo en la imagen
		(x, y, w, h) = [v * size_miniatura for v in face_i]

		# Obteniendo rostro
		#Ajustamos la imagen en un rectangulo
		face_img = frame[y:y+h, h:h+w].copy()

		#TODO: POR ENTEDER 
		blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

		#Ajustamos la imagen en un rectangulo
		face = gray[y:y + h, x:x + w]
		#rejustamos el tamaño de la imagen
		face_resize = cv2.resize(face, (im_width, im_height))

		# Intentado reconocer la cara
		prediction = model.predict(face_resize)

		#Dibujamos un rectangulo en las coordenadas del rostro
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
		
		#Ajustamos las imagenes en un rectangulo
		roi_gray = gray[y:y + h, x:x + w]
		roi_color = frame[y:y + h, x:x + w]
		
		#TODO: POR ENTEDER 
		eyes = eye_cascade.detectMultiScale(roi_gray)

		#Dibujamos un rectangulo en las coordenadas de los ojos
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

		#TODO: POR ENTEDER 
		#Predcir Genero
		gender_net.setInput(blob)
		gender_preds = gender_net.forward()
		gender = gender_list[gender_preds[0].argmax()]

		#TODO: POR ENTEDER 
		#Predecir Edad
		age_net.setInput(blob)
		age_preds = age_net.forward()
		age = age_list[age_preds[0].argmax()]

		# Escribiendo el nombre de la cara reconocida
		# La variable cara tendra el nombre de la persona reconocida
		cara = '%s' % (names_subdir[prediction[0]])

		#Si la prediccion tiene una exactitud menor a 100 se toma como prediccion valida
		if prediction[1]<100 :
			#Ponemos el nombre de la persona que se reconoció
			cv2.putText(frame,'%s : %.0f , %s , %s' % (cara,prediction[1], " Edad : "+str(age), "Gender:"+str(gender) ),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

			#En caso de que la cara sea de algun conocido se realizara determinadas accione          
			#Busca si los nombres de las personas reconocidas estan dentro de los que tienen acceso          
			flabs.valida_invitado(cara)

		#Si la prediccion es mayor a 100 no es un reconomiento con la exactitud suficiente
		elif prediction[1]>101 and prediction[1]<500:           
			#Si la cara es desconocida, poner desconocido
			cv2.putText(frame, 'Desconocido',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))  

		#Mostramos la imagen
		cv2.imshow('OpenCV Reconocimiento facial', frame)

	#Si se presiona la tecla ESC se cierra el programa
	key = cv2.waitKey(10)
	if key == 27:
		cv2.destroyAllWindows()
		break
