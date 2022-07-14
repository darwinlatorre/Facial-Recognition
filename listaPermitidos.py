#Clase de invitados
class flabianos:
	""" Lista de invitados"""

	#Lista de invitados reconocidos
	def __init__(self):
		self.Invitados=['luis','juan','pedro','carlos', 'darwin']

	#Valida si es un invitado registrado o no
	def valida_invitado(self,invitado):
		if invitado in self.Invitados:
			print('Bienvenido {}'.format(invitado))
		else:
			print('Lo siento {}, aun no trais el omnitrix'.format(invitado))