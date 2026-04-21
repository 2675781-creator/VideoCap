# Système Intelligent de Détection d'Obstacles Routiers

**VideoCap** est une solution IA conçue pour identifier et signaler en temps réel les obstacles routiers. En analysant des images issues de flux vidéo routiers, l'application permet de renforcer la sécurité en signalant la présence de véhicules, de piétons ou d'objets massifs.


## Fonctionnement du système
Entrée (Image) : Téléversement d'une image, d'une vidéo ou activation d'un flux webcam direct (formats supportés : .jpg, .jpeg, .png, .mp4, .avi)
Traitement (Analyse IA) : Le modèle détecte et classifie les obstacles en quatres catégories : vehicle, heavy_vehicle, human et cycle.
Sortie (Résultat annoté) : Affichage d'un flux annoté avec des boîtes englobantes autour des obstacles, accompagnée du nom de la catégorie et du score de confiance.


## Paramètres avancés
Pour maximiser l'efficacité du système, deux réglages sont à votre disposition dans la barre latérale :

-   Seuil de confiance :
    -   Réglage par défaut : 0.15.
    -   Permet de filtrer les détections. Un seuil bas maximise la sécurité en affichant même les obstacles incertains.

-   Optimisation du Contraste (CLAHE):
    -   Technologie "Vision Haute Visibilité".
    -   **À utiliser :** Par temps nuageux, dans les zones d'ombre ou pour les scènes à faible contraste afin de faire ressortir les piétons masqués.


## exemples d'utilisation :
a) Contexte urbain : Téléchargement d'un carrefour dense au format .jpg. Le système identifie instantanément les voitures et les cyclistes.

b) Scénario de sécurité : Image de validation montrant masqué par un camion est soumise. Le modèle détecte le piéton malgré l'occlusion partielle.



## erreurs fréquentes :
1. Téléversement d'une image dans un format non supporté (Erreur de format).
Solution : Convertir le fichier en format permit (.jpeg ou .mp4) par l'application avant de la soumettre.
2. Téléversement d'une image corrompue. Si l'image ne s'affiche pas, le fichier est peut-être endommagé. 
Solution : Téléchargez à nouveau l'image originale et réessayez.
3. Aucun objet détecté. Contraste trop faible ou seuil trop haut.
Solution : Activation de l'option **CLAHE** ou baissez le seuil de confiance à **0.10**.
4. Latence excessive dû à une résolution 
Solution : Sélectionnez une résolution d'inférence de **640px** dans les réglages.


Limites : 
1. **Seuil de confiance**: Un seuil de confiance est appliqué pour évité les fausses alertes (faux positifs). Mais certains objets hors contexte (feux rouges, panneaux) peuvent être brièvement confondus avec des silhouettes humaines.
2. **Conditions extrêmes**: La précision peut diminuer durant des conditions météo très dégradées (brouillard intense, nuit totale sans éclairage public)
3. **Objets rares**: Le modèle peut avoir des difficultés à identifier des objets très rares qui n'était pas présentes durant la phase d'apprentissage initial.


En cas de problème de l'utilisation de la plateforme, veuillez contacter l'équipe de développement : 

Daniel Bourcier Blake, 
Stephan Lo-Man, 
Elhadji Abdoulaye Diagne