# VideoCap - Détection d'obstacles routiers
VideoCap est une solution IA conçu pour améliorer la sécurité routière en détectant en temps réel les obstacles critiques (véhicules, piétons, cyclistes) dans des environnements dynamiques.


## Aperçu du projet
Le projet répond au défi de l'inattention au volant. L'objectif est de fournir une détection fiable, même pour les usagers vulnérables souvent négligés par les modèles standards.

**Le message clé :**La sécurité ne dépend pas seulement de la puissance du modèle, mais de la représentativité des données.

## Stratégie Data-Centric
Pour résoudre le déséquilibre sévère du dataset initial, nous avons mis en oeuvre un pipeline de curation massif : 

- **Curation sélective** : Passage de 200 images à 8311 images synchronisées.
- **Oversampling** : Multiplication par 5 des classes rares (humains, cycles).
- **Augmentations** : Utilisation de techniques de **Mosaïque** et **Mixup** pour renforcer la robustesse spatiale.

## Architecture Technique
- **Modèle**: YOLO11m (Medium) pour un équilibre optimal précision/vitesse.
- **Optimisation**: Taux d'apprentissage stabilisé (lr0 = 0.001), **Warmup** de 5 époques et gel des couches ("freeze=10").
- **Déploiement** : Interface utilisateur interactive via **Streamlit** et conteneurisation **Docker**.

## Installation et Utilisation

### Prérequis
- Docker & Docker Compose
- GPU NVIDIA (recommandé pour l'inférence rapide)

### Lancement via Docker
Bash :
**Cloner le dépôt**
"git clone https://github.com/2675781-creator/VideoCap.git
cd VideoCap

**Construire et lancer l'application Streamlit**
docker-compose up --build"

L'interface sera accessible sur "localhost:8501"

## Performance et Métriques
Le modèle vise les seuils critiques suivants pour une application insdustrielle :
- Rappel : >= 0.90 (Détection des piétons).
- Précision : >= 0.85 (Fiabilité des alertes).
- map-50 : >= 0.85 (Robustesse globale).

## Auteurs
- Daniel Bourcier Blake
- Stephan Lo-Man
- Elhadji Abdoulaye Diagne
