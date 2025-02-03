# Ciencia-de-Datos-Avanzada
Tarea 1.- Red Neuronal con SGD
Proyecto: Clasificación de daños en automóviles con redes neuronales convolucionales.

## Descripción del proyecto.

Esta tarea consiste en desarrollar un modelo de deep learning basado en redes neuronales convolucionales (CNN) para la clasificación de daños en automóviles. Esto a partir del conjunto de datos "Car Damage Severity Dataset" de Kaggle que contiene imágenes de vehículos con distintas clasificaciones de daños: menor, moderado y severo. A través de este modelo se busca analizar, procesar y aprender de imágenes de tal manera que logremos clasicar el grado de severidad con imágenes nuevas.

Este tipo de modelos de clasificación resultan muy útiles en la práctica, sobre todo en los giros de aseguradoras, ya que nos ayuda a mejorar la eficiencia a la hora de evaluar siniestros, resucir costos ya sea porque reduces el manejor de intermediarios como los son los ajustadores o se hacen precisiones más correctas en la hora del peritaje, así como ayuda a optimizar procesos de tomas de decisiones con base en fundamentos sólidos. El modelo CNN no solo nos ayuda a mejorar la precisión de nuestras clasificaciones, sino que también dirve para reducir tiempos en la inspección y evaluación del siniestro.

## Pasos necesarios para ejecutar el código.

Para llevar acabo la ejecución de este modelo es necesario partir de cersiorarnos de instalar las librearía necesarias, como lo son torch, torchvision, matplotlib y kaggle. Una vez instaladas las librerías, configuramos las credenciales mediantes las variables de entorno "kaggle_username" y "kaggle_key".

Seguimos con las descarga del conjunto de datos en Kaggle y descomprimimos creando carpetas separadas para las imágenes de entrenamiento y las de validación.

Continuamos con el preprocesamiento de datos, donde aplicamos transformaciones a la imágenes, como redimensionamiento a 224x224 pixeles, normalización y ajustes en la rotación, brillo y contraste de los datos para mejorar la generalización del modelo.

Una vez procesados los datos, definimos el modelo CNN construyendo una red convolucional de 4 capas, cada una seguida de normalización batch y una función de activación ReLu. Además, incluimos capas de max pooling para reducir la dimensionalidad de las imágenes de manera progresiva y una red completamente conectada (fully connected) por neuronas, seguida de una capa de salida con activación softmax para la clasifación de los 3 niveles por medio de probabilidades.
Luego, usamos un dropout con una probabilidad default de 0.5 para evitar sobreajuste y mejorar la generalización.

Para el entrenamiento del modelo, lo hacemos utilizando la función de pérdida "CrossEntropyLoss" y el optimizador SGD. Partimos de una tasa de aprendizaje y momentum para evitar que nuestro modelo oscile demasiado, adicional, con ayuda del StepLR hacemos que nuestro modolo se ajuste dinámicamente en la tasa de aprendizaje cada ciertas épocas.

Todo lo anterior se fija para ciertas épocas, donde en cada iteración el modelo realiza un backpropagation por medio de una función forward que declaramos, la cual nos ayuda a actualizar los pesos mediante el optimizador.



## Breve explicación del modelo implementado.
