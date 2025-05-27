# Курсова
Реалізація коду для курсової роботи

Для виконанання коду необхідні наступні бібліотеки:

- numpy
- matplotlib
- pickle
- collections
- scikit-learn
- pathlib
  
Рекомендована стурктура проєкту:
Проєкт

├── data

│   ├── BC

│   └── Control

└── src

    ├── cluster_analysis.py
    
    ├── colour_claster.py
    
    └── compute_similarity.py


В файлі compute_similarity.py в 46 рядку змінити шлях до папки data. Після цього можна запустити для виконання цей файл, а опісля файл cluster_analysis.py, який аналізує всю матрицю схожості. Файл colour_clusters.py можна запускати незалежно від інших, він аналізує кольорові канали по окремості. 
