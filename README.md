#Description
Ce projet vise à développer un système de sécurité intelligent capable de détecter le non-port des EPI dans un atelier de fabrication en temps réel, en utilisant des techniques avancées de vision par ordinateur et d'intelligence artificielle.

Fonctionnalités
Collecte et annotation des données : Création d'un dataset spécifique à partir d'images d'atelier, avec des annotations précises pour identifier les EPI.
Prétraitement des données : Nettoyage, transformation et data augmentation pour améliorer la qualité et l'équilibre du dataset.
Modélisation : Entraînement d'un modèle de détection basé sur YOLOv8, optimisé pour une détection en temps réel et robuste.
Déploiement : Intégration sur une carte STM32 pour un usage embarqué et interface web (HTML/CSS/JS) pour le suivi des alertes.
Technologies Utilisées
Frameworks et bibliothèques : PyTorch, YOLOv8, OpenCV
Langages : Python, HTML, CSS, JavaScript
Hardware : Carte STM32 pour déploiement embarqué
Outils de développement : Jupyter Notebook, outils d'annotation d'image
