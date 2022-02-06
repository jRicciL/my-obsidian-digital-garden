---
---

# Clase Modelo seq2seq

- Analisis de secuencias
Codificar --> Decodificar la informacion
- Variables latentes ==> Codificacion

Estados ocultos:
- Codificar toda la secuencia
![[Captura de Pantalla 2021-04-08 a la(s) 10.11.04.png]]

- Se puede hacer transferencia de aprendizaje usando el decodificador y haciendo fine tuning

- Los modelos LTSM no son capaces de almacenar en el estado oculto final toda la informacion:
	- Solo rescatan una pequenia parte del total de la informaci'on

### Aplicaciones
- El analisis de sentimientos es una alternativa de aplicaci'on de LSTM.
- --> Un ejemplo es el analisis de tendencias y opiniones en redes sociales como twitter

- Tambi'en para la generacion de resumenes


## Atencion
Articulo original:
- Neural machine translation by jointly learning to aling and translate
- Effectiove Approaches to Attention-based Neural Machine Translation

El utlimo estado de codificador se pasa directamente al decoder.
Pero el resto de las salidas de cada estado se ponderan y se pasan tambien al decoder...
--> ==Context Vector==
---> Atencion de Bahdanau
- > Determina a qu'e poner atencion
- Entrenamiento forzado en la parte del decoder, donde cada estado recibe el input ideal, que el estado previo tendria que haber arrojado
- ADem'as los estados ocultos se concatenan a travez del vector de contexto





### Tarea
#### Como mejorar el codificador LSTM

- podemos utilizar un dropout al definir el codificador lstm 
- En la tarea en lugar de trabajar a nivel caracter hay que trabajar a nivel palabra
- TOwards 
	- Intuitive undestanding atention mechanisms
	- Attention sequence 2 sequence model

##### Instrucciones
- Tokenizar debidamente las palabras
- necesitamos un corpus paralelo, que esten alineados
- no usamos el del ejemplo porque las frases son muy cortas y su implementacion por palabras puede que no sea tan factible
- Implementacion de word2vec en ingles a espanol... hay que descargarlo => es un diccionario, en terminos de python, que ya est'a indexado..
	- Hay que bajarlo en ingles y en espaniol
	- Luego tokenizar la expresion

- Implemetnar el modelo ose2seq para realizar el avance.

- Opus filter en github