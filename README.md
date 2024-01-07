# Learning Deep CNN Denoiser Prior for Image Restoration
Tutoriel sur les méthodes plug and play utilisation de réseaux de neurones pour élaborer des algorithmes d'optimisation convexe,  notamment pour les problèmes inverses (débruitage d'images, etc.).

- [lien de l'article](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Learning_Deep_CNN_CVPR_2017_paper.html)
- [lien du GitHub version matlab (version 2018)](https://github.com/cszn/ircnn)
- [lien du GitHub version PyTorch (version 2021)](https://github.com/cszn/DPIR)

Contenue du repo:
- `\cnn`: dossier contenant le code et les expériences du Denoiser. Voir `\cnn\README.md` pour plus d'information.
- `\hqs`: dossier contenant les codes pour faire la méthode hqs avec du plug and play (selon l'article). Procédure pour lancer le code:
  - Régler les paramètres `hqs\parameter.yaml` comme voulu.
  - Lancer `python hqs\hqs.py`
  - Retrouver vos résultats dans `\images`
- `\images`: contient tous les résultats enregistrés dans les différents sous-dossiers
- `\paper`: contient l'article, des notes de l'article, ...
- `\video`: contient les slides de la vidéo de l'article. Vous pouvez retrouver la vidéo dans sur le lien suivant: https://drive.google.com/file/d/14lCF0FoyuQ86Nv3bY_7Adnlule_ZAir1/view?usp=sharing

