# Ciencia-de-Datos-Avanzada
Tarea 1.- Red Neuronal con SGD
Proyecto: Clasificación de daños en automóviles con redes neuronales convolucionales.

## Descripción del proyecto.

Esta tarea consiste en desarrollar un modelo de deep learning basado en redes neuronales convolucionales (CNN) para la clasificación de daños en automóviles. Esto a partir del conjunto de datos "Car Damage Severity Dataset" de Kaggle que contiene imágenes de vehículos con distintas clasificaciones de daños: menor, moderado y severo. A través de este modelo se busca analizar, procesar y aprender de imágenes de tal manera que logremos clasicar el grado de severidad con imágenes nuevas.

Este tipo de modelos de clasificación resultan muy útiles en la práctica, sobre todo en los giros de aseguradoras, ya que nos ayuda a mejorar la eficiencia a la hora de evaluar siniestros, resucir costos ya sea porque reduces el manejor de intermediarios como los son los ajustadores o se hacen precisiones más correctas en la hora del peritaje, así como ayuda a optimizar procesos de tomas de decisiones con base en fundamentos sólidos. El modelo CNN no solo nos ayuda a mejorar la precisión de nuestras clasificaciones, sino que también dirve para reducir tiempos en la inspección y evaluación del siniestro.

## Pasos necesarios para ejecutar el código.

Para llevar acabo la ejecución de este modelo es necesario partir de instalar las librearía necesarias, como lo son torch, torchvision, matplotlib y kaggle. Una vez instaladas las librerías, configuramos las credenciales mediantes las variables de entorno "kaggle_username" y "kaggle_key" para así poder acceder a los datos.

Seguimos con las descarga del conjunto de datos en Kaggle y descomprimimos creando carpetas separadas para las imágenes de entrenamiento y las de validación.

Continuamos con el preprocesamiento de datos, donde aplicamos transformaciones a las imágenes para optimizar su procesamiento, como redimensionamiento a 224x224 pixeles para mantener un foramto que sea más uniforme y compatible al modelo, normalización y ajustes en la rotación, brillo y contraste de los datos para mejorar la generalización del modelo sobre datos nuevos, así como una mayor estabilidad en el entrenamiento.

Una vez procesados los datos, definimos el modelo CNN construyendo una red convolucional de 4 capas, cada una seguida de normalización batch que permite estabilizar el entrenamiento y acelarar la convergencia y una capa de max pooling para reducir la dimensionalidad de las imágenes de manera progresiva.

Además, se agrega una red completamente conectada (fully connected) por neuronas, seguida de una capa de salida para la clasifación de los 3 niveles por medio de probabilidades. Utilizamos activaciones ReLu en las capas "ocultas" para mejorar el aprendizaje y Softmax como una función de activación en la capa de salida.

Para el entrenamiento del modelo, lo hacemos utilizando la función de pérdida "CrossEntropyLoss" y el optimizador SGD. Partimos de una tasa de aprendizaje muy pequeña para ayudar al aprendizaje y momentum que ayude a evitar que nuestro modelo oscile demasiado. Adicional, con ayuda del StepLR hacemos que nuestro modolo se ajuste dinámicamente en la tasa de aprendizaje cada ciertas épocas, reduciendo la tasa de aprendizaje.

Todo lo anterior se fija para ciertas épocas, donde en cada iteración el modelo realiza un backpropagation por medio de una función forward que declaramos, la cual nos ayuda a actualizar los pesos mediante el optimizador. En este paso monitoremoas la pérdida y precisión en nuestros datos de entrenamiento.

Por último para la evaluación del modelo, evaluamos el conjunto prueba analizando las distintas métricas que solicitó el profesor, como lo fueron la precisión, recall, accuracy, f1-score y la matriz de confusión que nos ayude a tener mayor visibilidad en las clasificaciones certeras y las erradas, de manera que podamos buscar soluciones de mejoras o comportamientos irregulares.

## Breve explicación del modelo implementado.

En el modelo implementado utilizamos 4 capas convolucionales con tamaño de kernel 3x3 y un número progresivo de filtros: 32, 64, 128, 256. Aplicamos normalización batch después de cada convolución en busca de estabilizar y acelerar el aprendizaje.

Para las capas de max pooling buscamos reducir la dimesionalidad de las características quedándonos con las más relevantes y usando menor memoria, en nuestro modelo reducimos a la mitad al usar un stride de 2.

En las capas completamentes conectadas (fully connected) tenemos una capa de 512 neuronas que usan una activación ReLu para permitir capturar más combinaciones de características. Y luego tenemos, la capa de salidad que tiene 3 neuronas y usa activación Softmax, que ayuda a convertir las salidad en probabilidades para cada nivel de daño.

Para la regularización y optimización, utilizamos dropout con una probabilidad default de 0.5 para reducir el sobreajuste y mejorar la generalización del modelo en los nuevos datos. Para SGD utlizamos una tasa de aprendizaje de 0.0005 ya que vimos que al reducir mucho este rate, mejoraba el aprendizaje de nuestro modelo, también utilizamos un momentum de 0.9 que evita que los valores oscilen mucho.

Para el StepLR fijamos 20 épocas, por lo que se irá reduciendo la tasa de aprendizaje cada que cumpla con esta cantidad, lo que evitará que se estanque en mínimos locales.

Por último, entrenamos el modelo iterativamente a los largo de 50 épocas, dode en cada iteración calculamos las predicciones del modelo, medimos la pérdida, ajustamos los pesos nediante backpropagation en SGD, evaluamos el rendimiento y actualizamos la tasa de aprendizaje.
