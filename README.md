# Ciencia-de-Datos-Avanzada
Tarea 1.- Red Neuronal con SGD

Proyecto: Clasificación de daños en automóviles con redes neuronales convolucionales.
###### Equipo:
###### Ariana Escalante Herrera (216925)
###### Sarah Margarita López Ramos (216639)

## Descripción del proyecto.

Esta tarea consiste en desarrollar un modelo de deep learning basado en redes neuronales convolucionales (CNN) para la clasificación de daños en automóviles. Esto a partir del conjunto de datos "Car Damage Severity Dataset" de Kaggle que contiene imágenes de vehículos con distintas clasificaciones de daños: menor, moderado y severo. A través de este modelo se busca analizar, procesar y aprender de imágenes de tal manera que logremos clasificar el grado de severidad con imágenes nuevas.

Este tipo de modelos de clasificación resultan muy útiles en la práctica, sobre todo en los giros de aseguradoras, ya que nos ayuda a mejorar la eficiencia a la hora de evaluar siniestros, reducir costos ya sea porque reduces el manejo de intermediarios como lo son los ajustadores o se hacen precisiones más correctas en la hora del peritaje, así como ayuda a optimizar procesos de tomas de decisiones con base en fundamentos sólidos. El modelo CNN no solo nos ayuda a mejorar la precisión de nuestras clasificaciones, sino que también sirve para reducir tiempos en la inspección y evaluación del siniestro.

## Pasos necesarios para ejecutar el código.

Para llevar acabo la ejecución de este modelo es necesario partir de instalar las librearía necesarias, como lo son torch, torchvision, matplotlib y kaggle. Una vez instaladas las librerías, configuramos las credenciales mediantes las variables de entorno "kaggle_username" y "kaggle_key" para así poder acceder a los datos.

Seguimos con las descarga del conjunto de datos en Kaggle y descomprimimos creando carpetas separadas para las imágenes de entrenamiento y las de validación.

Continuamos con el preprocesamiento de datos, donde aplicamos transformaciones a las imágenes para optimizar su procesamiento, como redimensionamiento a 224x224 pixeles para mantener un formato que sea más uniforme y compatible al modelo, normalización y ajustes en la rotación, brillo y contraste de los datos para mejorar la generalización del modelo sobre datos nuevos, así como una mayor estabilidad en el entrenamiento.

Una vez procesados los datos, definimos el modelo CNN construyendo una red convolucional de 4 capas, cada una seguida de normalización batch que permite estabilizar el entrenamiento y acelerar la convergencia y una capa de max pooling para reducir la dimensionalidad de las imágenes de manera progresiva.

Además, se agrega una red completamente conectada (fully connected) por neuronas, seguida de una capa de salida para la clasifación de los 3 niveles por medio de probabilidades. Utilizamos activaciones ReLu en las capas "ocultas" para mejorar el aprendizaje y Softmax como una función de activación en la capa de salida.

Para el entrenamiento del modelo, lo hacemos utilizando la función de pérdida "CrossEntropyLoss" y el optimizador SGD. Partimos de una tasa de aprendizaje muy pequeña para ayudar al aprendizaje y momentum que ayude a evitar que nuestro modelo oscile demasiado. Adicional, con ayuda del StepLR hacemos que nuestro modelo se ajuste dinámicamente en la tasa de aprendizaje cada ciertas épocas, reduciéndola.

Todo lo anterior se fija para ciertas épocas, donde en cada iteración el modelo realiza un backpropagation por medio de una función forward que declaramos, la cual nos ayuda a actualizar los pesos mediante el optimizador. En este paso monitoreamos la pérdida y precisión en nuestros datos de entrenamiento.

Por último para la evaluación del modelo, evaluamos el conjunto prueba analizando las distintas métricas que solicitó el profesor, como lo fueron la precisión, recall, accuracy, f1-score y la matriz de confusión que nos ayuda a tener mayor visibilidad en las clasificaciones certeras y las erradas, de manera que podamos buscar soluciones de mejoras o comportamientos irregulares.

## Breve explicación del modelo implementado.

Al revisar las imágenes vimos que estas eran de distintos tamaños y entre todas las imágenes existían distintos patrones que complicaban la generalización del modelo, por esta razón decidimos hacer trasformaciones para los datos de entrenamiento y los datos de validación. 
Para el entrenamiento: Definimos un resize a 224x224 porque menores eran insuficientes para el detalle que se necesitaba para un mejor accuracy, aplicamos RandomHorizontalFlip que voltea la imagen horizontalmente con probabilidad de 0.5, RandomRotation que rota la imagen hasta 15 grados para que el modelo no dependa de la orientación, ColorJitter que cambia el brillo y constraste, RandomPerspective que aplica transformaciones de rotación y también de deformación, además, se convirtieron las imágenes a tensor y se normalizaron para mayor estabilidad.

Para los datos de validación, no se aplicaron todas las transformaciones, ya que lo que buscábamos era poder generalizar en la base de aprendizaje y capturar los patrones correctamente en los de prueba, por lo que solo se aplicó el redimensionamiento, se convirtieron a tensor y se normalizaron las imágenes.

En el modelo implementado utilizamos 4 capas convolucionales con tamaño de kernel 3x3 y un número progresivo de filtros: 32, 64, 128, 256. Aplicamos normalización batch después de cada convolución en busca de estabilizar y acelerar el aprendizaje.

Para las capas de max pooling buscamos reducir la dimesionalidad de las características quedándonos con las más relevantes y usando menor memoria, en nuestro modelo reducimos a la mitad al usar un stride de 2.

En las capas completamente conectadas (fully connected) tenemos una capa de 512 neuronas que usan una activación ReLu para permitir capturar más combinaciones de características. Y luego tenemos, la capa de salida que tiene 3 neuronas y usa activación Softmax, que ayuda a convertir las salidad en probabilidades para cada nivel de daño.

Para la regularización y optimización, utilizamos dropout con una probabilidad default de 0.5 para reducir el sobreajuste y mejorar la generalización del modelo en los nuevos datos. Para SGD utlizamos una tasa de aprendizaje de 0.0005 ya que vimos que al reducir mucho este rate, mejoraba el aprendizaje de nuestro modelo, también utilizamos un momentum de 0.9 que evita que los valores oscilen mucho.

Para el StepLR fijamos 20 épocas, por lo que se irá reduciendo la tasa de aprendizaje cada que cumpla con esta cantidad, lo que evitará que se estanque en mínimos locales.

Por último, entrenamos el modelo iterativamente a los largo de 50 épocas, dode en cada iteración calculamos las predicciones del modelo, medimos la pérdida, ajustamos los pesos mediante backpropagation en SGD, evaluamos el rendimiento y actualizamos la tasa de aprendizaje.

### Resultados:

Después de 50 épocas se obtuvo una exactitud de 64%, con una varianza relativamente baja. A diferencia de otras redes que se intentaron antes (con 2 y 3 capas, sin tantas tranformaciones, con mayor tasa de aprendizaje y sin dropout) vemos que el modelo no se encuentra excesivamente sobreajustado en el conjunto de entrenamiento. Así mismo, con menor cantidad de pixeles (128x128 pixeles) la precisión obtenida presentada mayor varianza entre las épocas.

Al analizar las métricas por clases podemos ver que:

1. Clase 0 (minor): obtuvo una precisión de casi 81%, es decir que de las veces que predijo esta clase acertó más del 80%, sin embargo el recall no fue tan alto como este con 71%.

2. Clase 1 (moderate): el modelo tiene problemas para identificar esta clase, la precisión fue la más baja, así como el F1-Score con 42%. Lo cual consideramos es congruente tomando en cuenta que es la clase intermedia.

3. Clase 2 (severe): esta clase obtuvo el recall más alto con 79%, lo que indica que de las veces que realmente eran choques severos si acertó en su mayoría de las veces en el modelo.

Se calcularon también las métricas generales del modelo, macro y ponderado, la primera se refiere a un promedio simple y la segunda es un promedio ponderado. Son muy similares ya que no existe un desbalanceo fuerte entre clases y observamos en la métrica ponderada un F1-Score de 62%, un recall de 62% y precisión de 63%.

Al observar la matriz de confusión vemos en la diagonal las predicciones correctas de cada clase y vemos el reflejo de las métricas antes mencionadas, confirmamos que la clase 1 es la más difícil de clasificar. En conclusión, vemos que el modelo reconoce bien la clase de choques menores y severos, sin embargo, le cuesta más trabajo distinguir a los choques moderados.

Derivado de los comentarios del Mtro. Mármol investigamos una aplicación "sencilla" de una red neuronal con la técnica de transfer learning para explorar una posible mejora a las métricas obtenidas hasta ahora.

Tranfer Learning reutiliza modelos preentrenados en tareas nuevas en lugar de entrenar un modelo desde cero como lo hicimos con la red convolucional.

## Ejercicio con Transfer Learning

Partimos de aplicar las mismas transformaciones mencionadas arriba. Aplicamos transfer learning con un modelo ResNet18.

Se carga el modelo ResNet18 que está preentrenado en ImageNet. 

Para mitigar el sobreajuste y acelerar el entrenamiento, se congelaron las primeras capas del modelo, ya que estas aprenden las características más generales y ya estaban preentrenadas en ImageNet. Además, utilizamos la función de pérdida "CrossEntropyLoss".

Para el modelo, utilizamos el optimizador Adam en lugar de SGD, ya que este tiene un mejor manejo cuando hay capas congeladas y porque ajusta automáticamente la tasa de aprendizaje de manera rápida y eficiente.

### Resultados:

Utilizando Transfer Learning logramos incrementar la precisión en la última época en 4.84 puntos porcentuales (68.95%) en comparación con la red convolucional entrenada desde cero. Sin embargo, se observa un fuerte sobreajuste en el conjunto de entrenamiento, a pesar de haber implementado un dropout de 0.6 como regularizador.

Como se mencionó anteriormente, para mitigar el sobreajuste se congelaron las primeras capas del modelo, no obstante, el sobreajuste sigue siendo significativo, lo que indica la necesidad de seguir ajustando la estrategia de entrenamiento.

En cuanto a las métricas por clase, la clase de choques menores obtuvo un mejor desempeño, con un F1-Score del 78.85%, manteniéndose como la categoría mejor clasificada. Le sigue la clase 2, mientras que la clase 1, aunque sigue siendo la más difícil de predecir, mejoró su F1-Score a 50% representando un aumento de 8 puntos porcentuales respecto al modelo anterior.

En la matriz de confusión observamos que el número de predicciones correctas (en la diagonal) fue mayor para todas las clases respecto al modelo anterior, excepto para la clase de choques severos, aunque el F1-Score aumentó de 70% a 75%.

En conclusión vemos que, a nivel general, este modelo de Transfer Learning con ResNet18 presentó un mejor rendimiento que la red convolucional, esto se debe a que aprovecha el conocimiento de modelos preentrenados, mientras que el primer modelo inicia desde cero. Sin embargo, hay un margen de mejora en nuestro modelo, la matriz de confusión muestra que una de las clases sigue siendo difícil de distinguir, lo que indica que podríamos explorar parámetros adicionales en nuestro modelo como ajustar más capas del modelo usando fine-tuning.

## Referencias:

NOTA: Las imágenes y gráficas mencionadas en el texto se encuentran en el código que se anexa.

1. Datt, K. (2023). Calculating output dimensions in a CNN for convolution and pooling layers with Keras. Medium. https://kvirajdatt.medium.com/calculating-output-dimensions-in-a-cnn-for-convolution-and-pooling-layers-with-keras-682960c73870
2. PyTorch. (n.d.). torch.nn.Conv2d. PyTorch Documentation. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
3. Towards Data Science. (2019). Conv2d: To finally understand what happens in the forward pass. Medium. https://medium.com/towards-data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
4. Hyperkai. (2020). RandomHorizontalFlip in PyTorch. DEV Community. https://dev.to/hyperkai/randomhorizontalflip-in-pytorch-57c3
5. GeeksforGeeks. (2021). Python PyTorch RandomHorizontalFlip function. GeeksforGeeks. https://www.geeksforgeeks.org/python-pytorch-randomhorizontalflip-function/
6. Hassan, E., Shams, M.Y., Hikal, N.A, y Elmoufy S. (2023). The effect of choosing optimizer algorithms to improve computer vision tasks: a comparative study. Multimed Tools Appl 82, 16591–16633. https://doi.org/10.1007/s11042-022-13820-0
