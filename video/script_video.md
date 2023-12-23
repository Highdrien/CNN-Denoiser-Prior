# vidéo

## Le problème
- parler du problème inverse $y=Hx+v$. explication du fait que l'on connait $H$.
- montrer les images x, H, et y
- expliquer comment on passe a cette equation: 
$$ \hat{x} = \text{arg} \min_x \frac{1}{2}||y-Hx||^2 + \lambda \Phi(x)  \quad (2)$$

## methode
- iSTA ET hqs
- implementer ISTA: constante de grandient: souvant 1 / constante de lipschizt. de $$|y-Hx||^2$$ la plus grande valeur singulère de $H$. (norme 2 de H).

## Les systemes de plug and play
- expliquer en quoi ça consiste
- expliquer ISTA
- expliquer la méthode HQS (et la diférance avec ISTA)

## Deep
- expliquer le model de deep learing proposé
- est ce qu'on parle des metrics (PSNR) ?
- regarder les noises

## Expérience
- montrer le Denoiser
- faire lancer les testes
- montrer les performances selon les itérations

## Montrer la v2
- montrer l'acticle v2 avec les améliorations (CNN -> UNET)
