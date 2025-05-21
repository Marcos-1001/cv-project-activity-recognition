

# Dataset

Las carpetas con los vídeos deben estar en el mismo nivel que el archivo de `feature_extraction.py`
https://www.kaggle.com/datasets/beosup/kth-human-motion?select=boxing



# Estructura 
- feature_extraction.py -> Lee todos los vídeos y extrae las características usando OpticalFlow Farneback y HOG en paralelo. Al final guarda un features.pkl que leerá el clasificador.  
- model.py -> Clasificador (SVM)

# Ejecución

- `make feature_extraction` para ejecutar la extracción de features
- `make model` para ejecutar el entrenamiento y evaluación de los features extraídos.
- `make both` para ejecutar los dos anteriores de manera secuencial. 



