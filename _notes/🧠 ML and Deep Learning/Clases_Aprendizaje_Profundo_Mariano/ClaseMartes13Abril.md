---
---

# Clase AP

## Seq2seq y atención

Familia de modelos que transforman secuencias a otras secuencias

- Posee estados internos
- H y C: Hidden y Carrier
- --> Estos son la entrada inicial al decodificador
- Los outputs se arrojan en cada time step: O_t
- A O_t se le aplica una capa densa para poder hacer la prediccion
- ![[Captura de Pantalla 2021-04-13 a la(s) 10.12.26.png]]

### Etapa de entrenamiento
Secuencia recorrida en el tiempo. Teacher forcing =>
- Utilizar la salida real como entrada en cada etapa o time step
- ![[Captura de Pantalla 2021-04-13 a la(s) 10.13.03.png]]

### Etapa de inferencia
![[Captura de Pantalla 2021-04-13 a la(s) 10.14.55.png]]

### Inconvenientes de Seq2Seq
![[Captura de Pantalla 2021-04-13 a la(s) 10.15.55.png]]


## Atencion
- Mecanismo para promoveer de memoria larga a los modelos Seq2Seq

- El foco puede mantenerse en distintas partes de la secuencia de entrada mientras es procesada, conserv'andose la informaci'on de contexto.

### Intuición

¿Cómo se propuso de forma inicial?

Traducir: *Atention is all you need*

- Se debe usar el contexto de la palabra ==> Estados ocultos

- Debemos cuantificar la influencia de cada palabra del contexto
- Se calcula un vector de atenci'on que indica que contexto influye m'as en cada palabra (o segmento de la secuencia)
![[Captura de Pantalla 2021-04-13 a la(s) 10.19.31.png]]
![[Captura de Pantalla 2021-04-13 a la(s) 10.21.06.png]]
![[Captura de Pantalla 2021-04-13 a la(s) 10.25.18.png]]
![[Captura de Pantalla 2021-04-13 a la(s) 10.27.21.png]]
![[Captura de Pantalla 2021-04-13 a la(s) 10.31.16.png]]
- El calculo no depende de forma unica del entrenamiento

### Como se implementa?

- En lstm usabamos unicamente el estado $H_{final}$, que servira como entrada para el decodificador


![[Captura de Pantalla 2021-04-13 a la(s) 10.36.07.png]]

![[Captura de Pantalla 2021-04-13 a la(s) 10.37.31.png]]

- Se utiliza una densa para todos los tiempos.

![[Captura de Pantalla 2021-04-13 a la(s) 10.39.45.png]]
- Tenemos dos h --> h_t (target) y h_s (source)

##### Operaciones 
- Attention weights
- Context vector
- Attention vector
- Hay dos estilos de multiplicacion para calcular el *score*
	- Luong's multiplicative style
- El vector de atenci'on ==> Vector alfa 
- Da como resultado el Vector de contexto 

### Etapa de Inferencia
![[Captura de Pantalla 2021-04-13 a la(s) 10.45.37.png]]

![[Captura de Pantalla 2021-04-13 a la(s) 10.48.59.png]]

```python
attention = 
```


# ImageNet
- **VGG16** => Problema de desvanecimiento del gradiente
- Alternativa, agragar batch normalization
- Normalizacion por lotes suele aplicarse previo a la funcion de activacion
	- Calcular la media y la varianza de los valores
	- Bloque convolucional 2d que conserva las dimensiones especiales
- Red Residual => Bloque de Identidad
	- Uso de puentas para saltar la fase de normalizacion y algunas de las capas
- Uso de bloques residuales
	- RESNET-34: Bloques de identidad
- UNet = 
	- Unet Convolutional networks for biomedical image segmentation
	- Dio origen a otro tipo de redes
	- Las capas procesadas se concatenan
	![[Captura de Pantalla 2021-04-13 a la(s) 11.14.34.png]]
	
	