🛰️ UrbanWatch – Détection de l’artificialisation urbaine à partir d’images Sentinel-2

UrbanWatch est un projet de data science et d’IA permettant de détecter automatiquement l’artificialisation urbaine à partir d’images satellite Sentinel-2 (résolution 10 m).
Il s’appuie sur :
un pipeline robuste de téléchargement & prétraitement des données SentinelHub,
la couche ESA WorldCover 2021 comme vérité-terrain,
un modèle Random Forest entraîné sur X millions de pixels,
une API FastAPI permettant de demander une prédiction d’artificialisation à n’importe quelle coordonnée GPS.


📌 Objectif

L’objectif du projet est de produire un modèle capable de prédire la probabilité d’artificialisation (bâti) à partir d’une tuile Sentinel-2 centrée autour d’un point GPS.
Le pipeline final permet :
de télécharger automatiquement les images satellite Sentinel-2,
d’appliquer un cloud-mask s2cloudless,
de calculer des indices spectraux (NDVI, NDBI, MNDWI…),
de standardiser et flatten chaque pixel en vecteur,
d’associer chaque pixel à sa classe ESA WorldCover (bâtis, eau, végétation…),
de transformer la tâche en binaire : urbain (=50) vs non-urbain,
d’entraîner un modèle Random Forest sur X millions de pixels,
de restituer une carte complète des prédictions du modèle et un score d'urbanisation moyen sur la zone,
d’exposer le tout dans une API permettant d’interroger le modèle.

📦 Architecture du projet

urban_watch/
```text
├── ml_logic/
│   ├── data.py               # Téléchargement SentinelHub, loading, metadata
│   ├── preprocessing.py      # Cloud mask, indices NDVI / NDBI / MNDWI, normalisation
│   ├── labels.py             # Conversion WorldCover, reprojection, cropping
│   ├── model.py              # Entraînement, prédiction, évaluation
│   ├── registry.py           # Tracking MLflow
│
├── interface/
│   ├── main.py               # Orchestration : full pipeline, training, prediction
│
├── api/
│   ├── api.py                # Serveur FastAPI pour prédictions en temps réel
│
├── data/
│   ├── features_x/           # Images Sentinel-2 téléchargées
│   ├── labels_y/             # Tuiles WorldCover reprojetées
│
├── requirements.txt
└── README.md
```

🚀 Méthodologie

1. 🛰️ Téléchargement Sentinel-2 (SentinelHub)
Pour chaque coordonnée GPS, une bbox 5 km × 5 km est générée, puis :
SentinelHubRequest (SENTINEL2_L2A)
Résolution : 10 m
MosaickingOrder : LEAST_CLOUD_COVERAGE
10 bandes Sentinel-2 récupérées (B01, B02, B03, … B12)
Les données brutes sont sauvegardées en .npy.

2. ☁️ Cloud masking (s2cloudless)
Un masque nuageux est généré et les pixels nuageux sont retirés.

3. 🧮 Calcul des indices spectraux
Trois indices essentiels sont ajoutés :
NDVI – végétation
NDBI – zones urbaines
MNDWI – eau et surfaces humides
→ L’image passe de 10 à 13 bandes.

4. ⚙️ Normalisation & flattening
Chaque image :
est normalisée bande-par-bande (min-max / standardisation),
est convertie en un tableau 2D de forme :
N_pixels_valides × 13 bandes

5. 🏷️ Construction des labels (WorldCover 2021)
Chaque bbox est convertie :
des CRS Sentinel-2 → WGS84,
découpée dans la tuile ESA correspondante,
reprojetée dans le CRS Sentinel-2 de la tuile X
(d’où l’apparition naturelle de quelques 0 = NoData qui sont à leur tour retirés sur y et sur X).
Les valeurs WorldCover sont ensuite converties :
50 = Built-up → 1
tout le reste → 0


6. 🤖 Modélisation : Random Forest
plus de X millions pixels utilisés pour l’entraînement
Features : 13 valeurs par pixel
Target : urbain vs non-urbain (binaire)
Le modèle final est enregistré sous MLflow, puis déployé dans l’API.


🔮 API FastAPI

L’API expose un endpoint permettant de :
téléchager une image Sentinel-2 autour d’un point GPS,
lancer le pipeline de preprocessing,
produire une prédiction,
renvoyer un score d’artificialisation moyen.
Endpoint principal
GET /predict?lon=5.4389&lat=43.5306&date=2021-06-15&size_km=3
Réponse :
```text
{
  "urbanization_score": 0.27,
  "prediction": [...300x300...],
  "image_rgb": [...encoded RGB...]
}
```

🧪 Reproductibilité

Installation
```text
pip install -r requirements.txt
```
Variables d’environnement
Créer un .env :
```text
SH_CLIENT_ID=xxx
SH_CLIENT_SECRET=xxx
BUCKET_NAME=...
GCP_PROJECT=...
```
Exécuter la pipeline complète
```text
from urban_watch.interface.main import full_preproc_pipeline
X, y = full_preproc_pipeline()
```
Entraîner le modèle
```text
from urban_watch.interface.main import train
train(model_name="random_forest_model")
```
Lancer l’API
```text
uvicorn api.api:app --reload
```


📈 Résultats

Le modèle Random Forest atteint :
```text
Précision : 
Recall : 
F1-score : 
Accuracy :
```
