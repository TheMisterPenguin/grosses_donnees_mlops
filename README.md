# Projet grosses données & mloups
Ange DESPERT  
Elias OKAT  
Camille REMOUE  

Pour démarre le projet il suffit de lancer cette commande depuis la racine du projet:
```bash
docker-compose up
```

## Architecture
Le projet est composé des services suivants:
- Un service grafana qui permet de visualiser les données (disponible sur le port 3001 (user: admin, password: admin))
- Un frontend qui permet d'appeler le modèle (disponible sur le port 3000)
- Un backend qui host le modèle
- Prometheus pour Grafana
- MongoExporter pour Prometheus

## Vidéo de présentation

[![Présentation du projet](https://img.youtube.com/vi/hA_pT3DkueU/0.jpg)](https://youtu.be/hA_pT3DkueU)