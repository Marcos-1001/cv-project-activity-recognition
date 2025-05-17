

# Dataset
https://www.kaggle.com/datasets/beosup/kth-human-motion?select=boxing

# Estructura 
- feature_extraction.py -> Lee todos los vídeos y extrae las características usando OpticalFlow Farneback. Al final guarda un features.pkl que leerá el clasificador 
- model.py -> Clasificador (SVM)

Si ejecutan el `model.py` con el features.pkl que he dejado les dará 72% de accuracy. Creo que aún se pueden hacer cosas para mejorar ese resultado. 



