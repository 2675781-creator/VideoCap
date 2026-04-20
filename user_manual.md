**Système Intelligent de Détection d'Obstacles Routiers**

**VideoCap** est une solution conçue pour identifier et signaler en temps réel les obstacles routiers. En analysant des images issues de flux vidéo routiers, l'application permet de renforcer la sécurité en signalant la présence de véhicules, de piétons ou d'objets massifs.

**Fonctionnement du système**
Entrée (Image) : Téléversement d'une photo routière provenant d'une caméra routière en formats (.jpg, .jpeg, .png)
Traitement (Analyse IA) : Le modèle identifie les véhicules, piétons et cyclistes.
Sortie (Résultat annoté) : Images affichant desc boîtes englobantes autour des obstacles, accmpagnée du nom de la catégorie et du score de confiance.

exemples :
a) Contexte urbain : Téléchargement d'un carrefour dense au format .jpg. Le système identifie instantanément les voitures et les cyclistes.

b) Scénario de sécurité : Image de validation montrant masqué par un camion est soumise. Le modèle détecte le piéton malgré l'occlusion partielle.


erreurs fréquentes :
1. Téléversement d'une image dans un autre format que (jpg, jpeg, png). 
Solution : Convertir le fichier en format permit par l'application avant de la soumettre.
2. Téléversement d'une image corrompue. Si l'image ne s'affiche pas, le fichier est peut-être endommagé. 
Solution : Téléchargez à nouveau l'image originale et réessayez.
3. Aucun objet détecté. 
Solution : S'assurez que l'image est suffisamment lumineuse ou utilisez une image de meilleure résolution.


Limites : 
1. **Seuil de confiance**: Un seuil de confiance est appliqué pour évité les fausses alertes (faux positifs).
2. **Conditions extrêmes**: La précision peut diminuer durant des conditions météo très dégradées (brouillard intense, nuit totale sans éclairage public)
3. **Objets rares**: Le modèle peut avoir des difficultés à identifier des objets très rares non présentes durant la phase d'apprentissage initial.


En cas de problème de l'utilisation de la plateforme, veuillez contacter l'équipe de développement : 

Daniel Bourcier Blake, 
Stephan Lo-Man, 
Elhadji Abdoulaye Diagne