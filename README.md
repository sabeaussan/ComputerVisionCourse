# Le projet SENTINEL

**Auteur :** samuel.beaussant@akkodis.com  
**Date :** 19 janvier 2026  

---

## Introduction

Bienvenue chez **SENTINEL** ! Vous venez d’être intégré à une startup innovante à fort impact en tant que développeur. Votre mission sera de développer un système de vision par ordinateur révolutionnaire permettant de détecter et de déclencher de manière automatique la maintenance d’infrastructures endommagées. SENTINEL utilise des drones afin de patrouiller de manière autonome au sein d’infrastructures critiques (ponts, bâtiments, barrages) pour inspecter leur intégrité structurelle. En cas de détection d’un défaut important, le système déclenche de manière automatique une opération de maintenance. Votre tâche est donc de la plus haute importance ! Nous comptons sur vous. Bonne chance et encore bienvenue chez SENTINEL !

**Disclaimer :**  
*Toute ressemblance avec un projet de recherche Akkodis ne serait que coïncidence fortuite.*

**Disclaimer 2 :**  
*En cas d’effondrement d’une infrastructure dû à un dysfonctionnement du module de vision, SENTINEL se réserve le droit de lancer des poursuites juridiques à votre encontre.*

---

## Mise en place

L’intégralité de la base de code actuelle de SENTINEL est contenue dans une archive à télécharger. Une fois que c'est fait, placez-vous dans le dossier de travail `course`.

L'environnement d'exécution est contenu dans un Docker afin de faciliter son déploiement. Vous allez donc devoir, dans un premier temps, le lancer.

Depuis un terminal, placez-vous dans le dossier de travail `course` puis tapez :

```bash
docker run --rm -it (--gpus all) --name vision_course -v PATH:/workspace -w /workspace sabeauss/vision_course bash
```

Sur Linux : `PATH="$PWD"`  
Sur Windows : `$PWD.Path` (PowerShell) ou `%CD%` (cmd)

L'argument optionnel `--gpus all` vous permet d'utiliser votre GPU CUDA-compatible à l'intérieur du container Docker.

Vous pouvez modifier les scripts Python depuis votre éditeur favori et les lancer depuis le Docker avec :

```bash
python3 script.py
```

Complétez les différents scripts avec le bon code.  
Votre carrière au sein de SENTINEL en dépend.

---

## Classification binaire avec DINO

Parmi les données récoltées, beaucoup ne contiennent pas de fissure. Notre équipe d’experts certifiés a effectué un premier travail de labellisation afin de classer une partie de ces images en deux groupes : positif (contient une fissure) et négatif (pas de fissure). Ils ont ensuite organisé les données comme suit :

- `train` : contient des exemples labellisés pour l’entraînement ;
- `val` : contient des exemples labellisés pour la validation ;
- `test` : contient des exemples **non labellisés**.

Par manque de temps, la plupart des données ne sont pas labellisées. Vous allez devoir créer un système pour faire le tri et garder uniquement les images contenant des fissures !

### Procédure

La famille de modèles DINO permet d’extraire des features très riches à partir d’images. Dans cette partie, nous allons entraîner un classifieur k-NN à partir de features extraites par un modèle DINO. Le k-NN sera entrainé sur `train`, validé sur `val` et permettra ensuite de classer les images `test`.

Le code à implémenter se trouve dans :
- `dino_knn.py`
- `dino_backbone.py`
- `io_utils.py`

Vous pouvez tester plusieurs valeurs de `k` et voir comment cela impacte les métriques. Quelle valeur de `k` choisir ? Vous pouvez aussi changer la version de DINO (v2 ou v3). À vous de jouer !

### Documentation pertinente (non exhaustive)

- https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- https://scikit-learn.org/stable/api/sklearn.metrics.html
- https://huggingface.co/docs/transformers/index
- https://docs.pytorch.org/docs/stable/index.html

---

## Segmentation avec SAM3

Les images étant maintenant triées, il convient de créer un système pour segmenter les fissures afin de pouvoir mesurer l’ampleur des dégâts. Malheureusement, à cause de multiples mouvements de grève, les experts certifiés de SENTINEL ne pourront pas effectuer la labellisation et produire les masques nécessaires pour entraîner un modèle de segmentation de fissures. Nous comptons donc sur votre ingéniosité !

### Procédure

L’absence de masques annotés nous empêche d’entraîner un modèle de segmentation supervisée classique. Nous utilisons donc SAM3 en segmentation à vocabulaire ouvert, ce qui permet de détecter et de segmenter des fissures sans phase d’apprentissage spécifique.

Le modèle est interrogé à l’aide du prompt textuel `"crack"`, qui lui demande de segmenter toutes les régions de l’image correspondant à des fissures. SAM3 retourne alors un ensemble de masques associés à des boîtes de détection.

Dans un premier temps, testez SAM3 sur l’image donnée dans le code. Vérifiez le résultat. Vous pouvez aussi tester le post processing avec Non Maximum suppression pour voir comment cela impacte le résultat ou modifier les paramètres de prediction. Pour finir, extrayez le mask de segmentation binaire. Vous pourriez avoir besoin de combiner plusieurs mask en un !

### Documentation pertinente (non exhaustive)

- https://docs.ultralytics.com/models/sam-3/
- https://docs.ultralytics.com/reference/__init__/
- https://docs.pytorch.org/docs/stable/index.html

---

## Segmentation avec YOLO

SENTINEL est très satisfait des résultats de votre système. Cependant, les contraintes matérielles et la taille du modèle en empêchent l’utilisation sur les drones d'inspection. SENTINEL tient à vous rappeler respectueusement que votre entretien mensuel de performance approche. À vous de trouver une solution !

### Partie 1

YOLO conviendrait parfaitement aux contraintes embarquées inhérentes aux drones. Il permettrait également de traiter un flux vidéo en temps réel sur CPU. Cependant, nous ne disposons pas de données labellisées pour l’entraîner sur de la détection de fissure. La dernière version de YOLO supporte l'inférence en vocabulaire ouvert mais les performances sont médiocres. Comment faire ?

### Partie 2

Maintenant que l’on a pu ajuster un modèle YOLO sur nos données, il est enfin temps de l’évaluer avant de le mettre en production ! À l’intérieur du fichier `eval_yolo.py`, écrivez le code pour effectuer son évaluation (mAP, recall). Une fois cette étape réussie, lancez l’inspection sur les données récoltées et identifiez les zones nécessitant une maintenance.

### Documentation pertinente (non exhaustive)

- https://docs.ultralytics.com/models/yolo26/#usage-example
- https://docs.ultralytics.com/reference/__init__/
- https://docs.pytorch.org/docs/stable/index.html

---

## Conclusion

Félicitations ! Vous êtes arrivé au terme de cette mission critique au sein de SENTINEL. Vous avez conçu et assemblé les briques fondamentales d’un système de vision par ordinateur moderne, capable de transformer des images brutes acquises par drones en décisions opérationnelles à fort impact.

L’équipe de SENTINEL vous remercie pour votre engagement. Grâce à vous, des infrastructures critiques pourront être surveillées de manière plus sûre, plus efficace et plus proactive. Malheuresement des coupures budgétaires nous obligent à mettre fin à votre contrat immédiatement. Cependant, SENTINEL recherche activement un stagiaire en vision par ordinateur et vous encourage à postuler (stage non rémunéré) ! En espérant vous revoir très bientot,

**L’équipe de SENTINEL**

![Vault Boy](Vault_Boy_artwork.png)
