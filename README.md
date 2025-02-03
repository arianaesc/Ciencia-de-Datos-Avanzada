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


