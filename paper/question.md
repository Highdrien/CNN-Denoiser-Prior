Analyse de l'article: Learning Deep CNN Denoiser Prior for Image Restoration

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Background](#background)
  - [Image Restoration with Denoiser Prior](#image-restoration-with-denoiser-prior)
  - [Half Quadratic Splitting (HQS) Method](#half-quadratic-splitting-hqs-method)
    - [Résolution de 6a (hors papier)](#résolution-de-6a-hors-papier)
    - [Suite du papier](#suite-du-papier)
- [Learning Deep CNN Denoiser Prior](#learning-deep-cnn-denoiser-prior)
  - [Why Choose CNN Denoiser?](#why-choose-cnn-denoiser)
  - [The Proposed CNN Denoiser](#the-proposed-cnn-denoiser)
- [Experiments](#experiments)
  - [Image Denoising](#image-denoising)
  - [Image Deblurring](#image-deblurring)
  - [Single Image Super-Resolution](#single-image-super-resolution)
- [Conclusion](#conclusion)

# Abstract

2 méthodes pour faire de IR (Image Restoration):
- model-based optimization
- discriminative learning methods

> Le but de l'article est de *plug in* un CNN (qui est un discriminative learning method) devant un model-based optimizarion pour avoir des meilleures performances.

# Introduction

On dégrade une image $x$ avec $y=Hx+v$, où $H$ est une matrice de dégradation, $v$ un bruit gaussien de std $\sigma$.

- *image denoising* quand $H=I_n$
- *image deblurring* quand $H$ est un opérateur de floutage
- *image super resolution* quand $H$ est la composition d'un opérateur de floutage et un down-sampling.
  
> Le but est de résoudre le problème inverse, c'est à dire de trouver $x$ connaissant $y$ et $H$. Pour cela on veut maximiser $\max_x \log p(x|y) = \max_x \log p(y, x)$

D'un point de vue Bayesian, on peut résoudre $\hat{x}$ avec Maximum A Posterior (MAP):
$$ \hat{x} = \text{arg} \max_x \log p(y|x) + \log p(x)  \quad (1)$$

avec:
- $\log p(y|x)$: log-likelihood of observation $y$
- $\log p(x)$: the prior of $x$ (independent of $y$)


L'équation est équivalente à:
$$ \hat{x} = \text{arg} \min_x \frac{1}{2}||y-Hx||^2 + \lambda \Phi(x)  \quad (2)$$

L'idée est de résoudre cette nouvelle équation avec un modèle de deep learning et minimiser la loss $l$:
$$ \min_{\Theta} l(\hat{x}, x) \textit{ s.t. } \hat{x} = \text{arg} \min_x \frac{1}{2}||y-Hx||^2 + \lambda \Phi(x, \Theta)  \quad (3)$$

Ils vont remplacer MAP par un CNN avec several CNN techniques, including Rectifier Linear Units (ReLU), batch normalization, Adam, dilated convolution. Puis plugged in as a modular part of model-based optimization methods.

# Background

## Image Restoration with Denoiser Prior

Plusieurs tentatives ont été faites pour incorporer un débruiteur dans des méthodes d'optimisation basées sur des modèles pour d'autres problèmes inverses:
- *BM3D frames and variational image deblurring* qui utilise l'équilibre de Nash
- *single image super-resolution (SISR)* qui utilise la méthode CBM3D denoiser qui a des meilleures performances sur la métrique PSNR par rapport à la méthode SRCNN.
> PSNR (Peak Signal to Noise Ratio) est une mesure de distorsion utilisée en image numérique, tout particulièrement en compression d'image. Elle permet de quantifier la performance des codeurs en mesurant la qualité de reconstruction de l'image compressée par rapport à l'image originale. $PSNR=10\times\log_{10}(\frac{255^2}{MSE})$.
- de nombreuses autres méthodes 

Finalement, ils vont choisir la HQS Method pour sa simplicité. Elle peut faire une multitude de chose, notamment être incorporée dans le denoiser prior.

## Half Quadratic Splitting (HQS) Method

> idée:  diviser la variable pour découpler le terme de fidélité et le terme de régularisation.

Avec cette idée on ré-écrit l'équation comme ceci:
$$ \hat{x} = \text{arg} \min_x \frac{1}{2}||y-Hx||^2 + \lambda \Phi(z) \quad s.t. \quad  z=x \quad (4)$$

Le but de la HSQ est donc de résoudre le problème, où $\mu$ est un penalty parameter:
$$ \mathcal{L}_{\mu}(x,z)=\frac{1}{2}||y-Hx||^2+\lambda \Phi(z) + \frac{\mu}{2}||z-x||^2 \quad (5)$$


On peut résourdre l'équation (5) par itération:
$$ x_{k+1}= \text{arg} \min_x ||y-Hx||^2+ \mu||x-z_k||^2 \quad (6a)$$
$$ z_{k+1}= \text{arg} \min_z \frac{\mu}{2}||z-x_{k+1}||^2+ \lambda \Phi(z) \quad (6b)$$

> On peut démontrer ça en remarquer que $L(\cdot, z) et L(x, \cdot)$ sont décroissante, donc par itération, on converge vers un minimum.

En fait l'équation (6a) peut être résolue directement par:
$$ x_{k+1} = (H^T H + \mu I)^{-1} (H^T y + \mu z_k) \quad (7)$$

> Annulation du gradient de l'équation (6a)

> En fait dans les fait, l'équation (7) est réalisable seulement si H est petit. Dans notre cas, c'est pas le mieux.

### Résolution de 6a (hors papier)

On veux minimizer (6a). On calcule alors le gradient et on l'annule avec $z_k$ fixé:
$$-2H^T(y-Hx)+2\mu(x-z_k)=0$$
$$\Leftrightarrow Gx=u$$
où $G=H^TH+\mu I$ et $u=H^Ty+\mu z_k$. L'idée est de ne pas inverser $G$ mais d'utiliser une méthode d'approximation pour trouver $x$.

**Autre idée**: Utiliser fourier (en disans que $H$ est une convolution) et avoir:
$$(H^TH+\mu I)^{-1}x \leftrightarrow (|H(\xi)|^2+\mu)^{-1}X(\xi)$$

### Suite du papier

Et on peux ré-écrire l'équation (6b) comme ceci (c'est seulement une division par $\lambda$):
$$ z_{k+1} = \text{arg} \min_z \frac{1}{2(\sqrt{\lambda / \mu})^2} ||x_{k+1}-z||^2 + \Phi(z) \quad (8) $$

Et cette équation peut être vue comme le denoiser d'une image $x_{k+1}$ avec un bruit gaussien de $\sqrt{\lambda / \mu}$.
Du coup, on a:
$$z_{k+1} = Denoiser(x_{k+1}, \sqrt{\lambda / \mu}) \quad (9)$$

Pour passer de l'équation (8) à (9), ils ont implicitement remplacé $\Phi(\cdot)$ par une priorité de débruitage. Cela offre plusieurs avantages:
- Premièrement, elle permet d'utiliser n'importe quel débruiteur en niveaux de gris ou en couleur pour résoudre divers problèmes inverses.
- Deuxièmement, la priorité d'image explicite $\Phi(\cdot)$ peut être inconnue lors de la résolution de l'équation (2).
- Troisièmement, plusieurs débruiteurs complémentaires exploitant différentes priorités d'image peuvent être utilisés conjointement pour résoudre un problème spécifique.

# Learning Deep CNN Denoiser Prior

## Why Choose CNN Denoiser?

Etat de l'art des Denoiser:
-  total variation (TV)
-  Gaussian mixture models (GMM) 
-  K-SVD
-  non-local means
-  BM3D

Mais ils ont tous des inconvénients, notamment leurs rapidités d'exécutions. Il faut aussi prendre en compte que dans la grande majorité des cas, les images seront dans le format RGB, et donc par conséquent, il est préférable de prendre une méthode qui est performante sur ce format. Ils ont donc finalement choisi les CNN pour le denoiser. Ce choix a notamment été motivé par 4 raisons:
- rapidité de l'inference (grâce notamment à la parallélisation sur GPU)
- il a une grande capacité de modéliser une prior avec son deep architecture
- external prior (CNN) est complémentaire à internal prior (comme BM3D)
- Les progrès des entraînements et du design de l'architecture des CNN vont faciliter leurs utilisations.

## The Proposed CNN Denoiser

Architecture du CNN utilisé:
<p align="center"><img src=model.png><p>

It consists of **seven** layers with three different blocks, i.e., “Dilated Convolution+ReLU” block in the first layer, five “Dilated Convolution+Batch Normalization+ReLU” blocks in the middle layers, and “Dilated Convolution” block in the last layer. The **dilation factors** of
(3×3) dilated convolutions from first layer to the last layer
are set to 1, 2, 3, 4, 3, 2 and 1, respectively. The number
of **feature maps** in each middle layer is set to 64.

Ils ont utlisé des **Dilated Filter** (filtre dilatés) car cela étend le domaine visible par le filtre.

Ils ont utlisé de la **Batch Normalization** et des **Residual Learning** car cela accélère l'entraînement.

Ils ont utilisé des images de petite taille dans leur entraînement, ce qui limite les effets de bords (boundary artifacts). Ils ont mis du 0-padding dans leurs convolutions. Et ils ont remarqué (empiriquement) qu'avec des petites images, ils avaient moins d'effet de bord. La raison est que si l'on fait des patches plus petit, on va avoir plus d'image et donc le CNN va voir plus de bords. Il sera donc plus entrainé à gérer les bords, et donc à essayer au maximum d'éviter les effets de bord. Cependant, il faut quand même que la taille de l'image soit plus grande que "receptive
field" des CNNs. Ils ont donc pris une taille d'image de $35\times35$ sans chevauchement.

Ils ont utilisé un faible **noise level** sur les images. Ils ont entrainé 25 denoiser sur des images avec un niveau de gris entre 0 et 50 (chacun avec un pas de 2).

> Je me suis arrêté là pour l'instant.

# Experiments

## Image Denoising

## Image Deblurring

## Single Image Super-Resolution

# Conclusion