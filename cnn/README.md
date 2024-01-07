# Entrainement d'un Denoiser

## Data

les données utilisées sont issus de la base de données: https://data.vision.ee.ethz.ch/cvl/DIV2K/

Le dossier `data` contient:
- `/DIV2K`: contient un ensemble de 900 images en couleur en bonne qualité (2K)
- `create_patches.py`: qui est un script python qui fait des patches de tailles $256 \times 256$
- `/center_patches` qui contient 900 images, obtenue en prennant le centre de chaque images de `/DIV2K` avec une taille de $256 \times 256$.
- `/all_patches` qui contient est le résultat du découpage de chaque image de `/DIV2K` avec une découpe de $256 \times 256$, sans chevochement.

## Modèle

Le modèle utilisé est une suscétion de CNN avec des filtres $3\times3$ et diférentes taille de dilatation. Le modèle proposé par l'article contient la sucésion $[1, 2, 3, 4, 3, 2, 1]$ (de taille de dilatation). J'ai entraîné un modèle un peu plus petit avec $[1, 2, 3, 2, 1]$. Ce modèle se nome: `little_model`.

## Utilisation

Se mettre dans `\cnn` et faire
```bash
python main.py --mode <yout_mode> --path <logs/your experiment folder>
```

Vous avez les modes suivant:
- `train`: sert à entrainer un nouveau modèle, selon la config: `\config\config.yaml`. Va créer un dossier dans `\logs` avec les poids du modèle, les métriques, ... Il n'y a pas besoin de spécifier un `--path`
- `test`: fait passer un modèle déjà entraîner sur la base de données de test. \\ Exemple: `path main.py --test --path logs/cosntant` pour tester le modèle constant.
- `bluring`: pour créer les images bruitées selon une config. Crée un dossier `..\images\<experiment_name>_blured` avec les images bruitées.
- `infer`: fait l'inférence sur les images bruitées. Crée un dossier `..\images\<experiment_name>_infer` avec les images débruitées.

