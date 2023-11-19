---
---


# Grad-CAM: Visualizacion de mapas de activación

Basado en el [ejemplo de Keras](https://keras.io/examples/vision/grad_cam/)

Mariano Rivera

Junio 2020

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.transform import resize
from PIL import Image

import tensorflow as tf
import tensorflow.keras as keras
```

Al llamar evaluar el modelo, en TensorFlow 2.1-2.2 se produce un error que se espera sea resuelto en próximas liberaciónes de la librería:

`UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.`

Mientras tanto, es necesario incluir las siguientes líneas justo después de vargar tensorflow y antes de definir el modelo (o leerelo):

```python
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
```

Tenemos las siguientes opciones para imágenes de prueba.

```python
local_image=False
if local_image:
    filenames=['african_elephant.png', 'autonomous_mex.png', 'beisbol.png', 'horses.png','rojas_at_angel.png', 'metro.png', 'rx_quico.png']
    filename=filenames[0]
else:
    from PIL import Image
    from urllib.request import urlopen
    url = 'https://i.pinimg.com/originals/2f/cf/d8/2fcfd89250f0774daae19e65346b2706.jpg'
    
    url = 'https://s3fs.bestfriends.org/s3fs-public/Introduce-cat-dog-Cappuccino-6654sak.jpg'
    filename = urlopen(url)

print(filename)
```

```
<http.client.HTTPResponse object at 0x7f478c104ee0>
```

También probamos entre 3 modelos distintos de red

```python
nombres_modelos= ['xception', 'vgg16', 'efficientnet']
seleccion=nombres_modelos[1]
```

## Modelo Xception

```python
if seleccion == nombres_modelos[0]:
    from tensorflow.keras.applications import Xception

    scale    =  255
    img_size = (299,299,3)

    model = Xception(input_shape = img_size,
                     include_top = True,
                     weights     = 'imagenet')

    preprocess_input   = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions

    last_conv_layer_name   = "block14_sepconv2_act"
    classifier_layer_names = ["avg_pool", "predictions",]

    #model.summary()
```

## Modelo VGG16

```python
if seleccion == nombres_modelos[1]:

    from tensorflow.keras.applications import VGG16

    scale=1
    img_size  = (224,224,3)

    model = VGG16(input_shape = img_size,
                  include_top = True,
                  weights     = 'imagenet')

    preprocess_input   = keras.applications.vgg16.preprocess_input
    decode_predictions = keras.applications.vgg16.decode_predictions

    last_conv_layer_name   = 'block5_conv3'
    classifier_layer_names =  ['block5_pool', 'flatten', 'fc1', 'fc2',"predictions",]

    #model.summary()
```

## Modelo EfficientNet B0

```python
if seleccion == nombres_modelos[2]:

    import tensorflow.keras as keras
    import efficientnet.tfkeras as efn 
    scale=255
    img_size = (224,224)
    model = efn.EfficientNetB0(weights='imagenet') 

    last_conv_layer_name   = 'top_activation'
    classifier_layer_names =  ['avg_pool', 'top_dropout', 'probs']

    #model.summary()
```

```python
def get_img_array(img_path, img_size):
    
    img = Image.open(img_path)
    
    img = img.resize(size=img_size)
    print(f'format: {img.format}, shape: {img.size}, mode: {img.mode}')
    img_array = np.array(img).astype('float32')[:,:,:3]  # tiramos el canal alpha
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array
```

```python
# load and show an image with Pillow
#'african_elephant.png',
img,img_array =  get_img_array(img_path = filename,    
                               img_size = img_size[:2])

img_array = img_array/scale
plt.imshow(img)
plt.axis('off')
plt.show()
```

```
format: None, shape: (224, 224), mode: RGB
```

![png](http://personal.cimat.mx:8181/~mrivera/cursos/aprendizaje_profundo/GradCAM/output_16_1.png)

```python
# Imprimir las predicciones mas altas, seleccionaremos la más alta
K = 5
preds = model.predict(img_array)
decoded_predictions=decode_predictions(preds, top=5)[0]

print("{:10} {:20} {:10}".format('Id. clase', 'Nombre', 'Probabilidad'))
print(22*' -')
for decoded in decoded_predictions:
    print("{:10s} {:25s} {:0.5}".format(decoded[0], decoded[1], decoded[2]))
```

```
Id. clase  Nombre               Probabilidad
 - - - - - - - - - - - - - - - - - - - - - -
n02124075  Egyptian_cat              0.45412
n04209239  shower_curtain            0.055721
n02110185  Siberian_husky            0.046105
n02109961  Eskimo_dog                0.041386
n02091467  Norwegian_elkhound        0.036881
```

```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    
    # Modelo que mapea la imagen de entrada a la capa convolucional última,
    # donde se calculará la activación
    last_conv_layer  = model.get_layer(last_conv_layer_name)
    conv_model       = keras.Model(model.inputs, last_conv_layer.output)

    # Modelo que mapea las activaciones a la salida final
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)
    
    # Cálculo del gradiente la salida  del modelo clasificador respecto a     
    with tf.GradientTape() as tape:
        
        # Calcula activacion del modelo base convolucional
        last_conv_layer_output = conv_model(img_array)
        tape.watch(last_conv_layer_output)
        
        # Calcula la predicción con modelo clasificador, para la clase mas probable
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        print(top_pred_index)
        top_class_channel = preds[:, top_pred_index]

    # Obtenemos el gradiente en la capa final clasificadora con respecto a
    # la salida del modelo base convolucional
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # Vector de pesos: medias del gradiente por capas,
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # salida de la última capa convolucional
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    
    # saliencia es la respuesta promedio de la última capa convolucional
    saliency = np.mean(last_conv_layer_output, axis=-1)
    saliency = np.maximum(saliency, 0) / np.max(saliency)
    
    # Multiplicación de cada canal por el vector de pesos
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
        
    # Heatmap: promedio de cada canal por su peso
    grad_cam = np.mean(last_conv_layer_output, axis=-1)
    grad_cam = np.maximum(grad_cam, 0) / np.max(grad_cam)
    
    return grad_cam, saliency
```

```python
# Generate class activation heatmap
grad_cam, saliency = make_gradcam_heatmap(img_array, 
                                          model, 
                                          last_conv_layer_name, 
                                          classifier_layer_names)
```

```
tf.Tensor(285, shape=(), dtype=int64)
```

```python
def show_hotmap (img, heatmap, title='Heatmap', alpha=0.6, cmap='jet', axisOnOff='off'):
    '''
    img     :    Image
    heatmap :    2d narray
    '''
    resized_heatmap=resize(heatmap, img.size)
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(resized_heatmap, alpha=alpha, cmap=cmap)
    plt.axis(axisOnOff)
    plt.title(title)
    plt.show()
```

```python
plt.subplot(121)
plt.imshow(grad_cam, 'jet')
plt.title('GradCam')
plt.subplot(122)
plt.imshow(saliency, 'jet')
plt.title('Saliencia')
plt.show()
```

![png](http://personal.cimat.mx:8181/~mrivera/cursos/aprendizaje_profundo/GradCAM/output_21_0.png)

### Sobreposición de los mapas de calor en la imagen original

```python
show_hotmap(img=img, heatmap=grad_cam, title=f'Grad Cam: {model.name}')
```

![png](http://personal.cimat.mx:8181/~mrivera/cursos/aprendizaje_profundo/GradCAM/output_23_0.png)

```python
show_hotmap(img=img, heatmap=saliency, title=f'Saliencia: {model.name}')
```

![png](http://personal.cimat.mx:8181/~mrivera/cursos/aprendizaje_profundo/GradCAM/output_24_0.png)

Note como, GradCam efectivamente muestra el mapa de activación para la clase seleccionada, en nuestro ejemplo seleccionamos automáticamente la de mayor probabilidad. En tanto la saliencia muestra los mapas de calor para las clases mas probables.