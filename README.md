# Le projet SENTINEL

**Auteur :** samuel.beaussant@akkodis.com  
**Date :** 19 janvier 2026  

---

## Introduction

Bienvenue chez **SENTINEL** !

Vous venez d’être intégré à une startup innovante à fort impact en tant que développeur. Votre mission sera de développer un système de vision par ordinateur révolutionnaire permettant de détecter et de déclencher de manière automatique la maintenance d’infrastructures endommagées.

SENTINEL utilise des drones afin de patrouiller de manière autonome au sein d’infrastructures critiques (ponts, bâtiments, barrages) pour inspecter leur intégrité structurelle. En cas de détection d’un défaut important, le système déclenche automatiquement une opération de maintenance.

Votre tâche est donc de la plus haute importance !  
Nous comptons sur vous. Bonne chance et encore bienvenue chez SENTINEL !

> **Disclaimer :**  
> Toute ressemblance avec un projet de recherche Akkodis ne serait que coïncidence fortuite.  
>
> **Disclaimer 2 :**  
> En cas d’effondrement d’une infrastructure dû à un dysfonctionnement du module de vision, SENTINEL se réserve le droit de lancer des poursuites juridiques à votre encontre.

---

## Mise en place

L’intégralité de la base de code actuelle de SENTINEL est contenue dans une archive à télécharger. Une fois que c'est fait, placez-vous dans le dossier de travail `course`.

L'environnement d'exécution est contenu dans un **Docker** afin de faciliter son déploiement.

```bash
docker run --rm -it (--gpus all) --name vision_course \
-v PATH:/workspace -w /workspace sabeauss/vision_course bash
```

---

## Classification binaire avec DINO

Organisation des données :
- `train`
- `val`
- `test`

Objectif : trier les images contenant des fissures à l’aide de DINO + kNN.

---

## Segmentation avec SAM3

Segmentation à vocabulaire ouvert à l’aide du prompt `"crack"` pour extraire des masques de fissures.

---

## Segmentation avec YOLO

Adaptation d’un modèle YOLO aux contraintes embarquées et évaluation via mAP et recall.

---

## Conclusion

Merci pour votre contribution au projet SENTINEL.

![Vault Boy](Vault_Boy_artwork.png)
