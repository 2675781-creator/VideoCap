Le but de notre solution est de pouvoir détecter et identifier en temps réel les obstacles routiers. En analysant des images issues de flux vidéo routiers, l'application permet de renforcer la sécurité en signalant la présence de véhicules, de piétons ou d'objets massifs.

Utilisation :
Entrée : Image routier provenant d'une caméra routière en formats (.jpg, .jpeg, .png)
Sortie : Images avec boîté englobantes : tracées autour de chaque obstacles détecté. Le nom de la classe et le score de confiance.

exemples :
a) Contexte urbain : Téléchargement d'un carrefour dense au format .jpg. Le système identifie instantanément les voitures et les cyclistes.

b) Scénario de sécurité : Image de validation montrant masqué par un camion est soumise. Le modèle détecte le piéton malgré l'occlusion partielle.


erreur fréquentes :
1. Téléversement d'une image dans un autre format que (jpg, jpeg, png). Solution : Convertir l'image en format permit par l'application.
2. Téléversement d'une image corrompue. Solution : Téléchargez à nouveau l'image originale et réessayez.
3. Aucun objet détecté. Solution : S'assurez que l'image est bien éclairée ou augmentez la résolution.


Limites : 
1. Seuil de confiance limité pour évité les faux positifs.
2. Formats de fichiers seulement pour (jpg, jpeg, png)
3. LA précision peut diminuer durant des conditions météo très dégradées (brouillard intense, nuit totale sans éclairage public)
4. Le modèle peut avoir des difficultés avec des classes très rares non présentes dans le dataset initial.


En cas de problème de l'utilisation de la plateforme, veuillez contacter les personnes suivantes : 

Daniel Bourcier Blake, 
Stephan Lo-Man, 
Elhadji Abdoulaye Diagne