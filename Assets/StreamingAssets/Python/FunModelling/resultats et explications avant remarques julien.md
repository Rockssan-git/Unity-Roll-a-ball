# Roll-a-Ball : résultats des tests et questions

Fichier testé : `roll_a_ball_progress.py`, à placer à la racine du dépôt et à lancer avec `python roll_a_ball_progress.py`

## Fonctionnement

- **Bloc D** : choisit où aller, à une distance d'au plus 1 de la balle. Il apprend une valeur V(s) par TD(0) avec un GP hétéroscédastique : chaque cible TD est ajoutée avec son temps global, et les échantillons anciens deviennent plus bruités. Réglages : wr = 3, mémoire de 200, γ = 0,95, lr = 0,5.
- **Bloc M** : choisit la commande qui amène la balle là où D veut aller. Son contexte est la position de la balle.
- **Récompense** = objectifs ramassés pendant le coup + `PROGRESS_WEIGHT` × rapprochement de l'objectif le plus proche.
- Le bloc D choisit la cible qui maximise (rapprochement immédiat × poids + γ·V(état suivant)), ce qui revient à une prédiction de la cible TD. Si on maximise V seul, la balle part loin des objectifs, parce que V est plus élevée là où il reste plus de chemin à gagner.

## Résultats

### Niveau facile : 3 objectifs alignés en (1,1), (2,2), (3,3), 6 coups par partie, 30 parties

Objectifs ramassés par partie (sur 3), moyenne sur 3 graines :

| Agent | 10 premières parties | 10 dernières parties |
|---|---|---|
| Hasard (commandes aléatoires) | — | 0,07 |
| Sans bonus (poids 0) | ≈ 0 | 0,3 à 0,5 |
| Poids 0,1 | 2,3 | 1,3 |
| Poids 0,2 | 2,4 | 1,8 |
| Poids 0,5 | 2,5 | 2,2 |
| Poids 1 | 2,5 | 2,7 |

### Niveau dur : 6 objectifs éparpillés, arène de -3 à 4, poids 1, une seule graine

| Coups par partie | Partie 1 | Parties suivantes |
|---|---|---|
| 10 | 1/6 | entre 4/6 et 5/6, sans tendance nette |
| 16 | 2/6 | 6/6 dès la partie 2 |

### Constats

1. **Sans bonus**, la récompense est rare : l'agent ne fait que légèrement mieux que le hasard en 30 parties. Tant qu'il n'a rien ramassé, V reste plate.
2. **Avec bonus**, l'agent joue presque parfaitement dès la 2e partie. C'est surtout le bloc M qui apprend, pendant la 1re partie. Le bloc D suit le bonus, ce qui revient à « va vers l'objectif le plus proche », et dans ces niveaux sans obstacle cette stratégie suffit.
3. **Avec un petit bonus** (0,1 ou 0,2), le score **baisse** au fil des parties. Une fois que V a appris, le bonus s'annule en grande partie dans le choix de D (c'est la propriété connue du reward shaping par potentiel). D se retrouve alors guidé par ce que V a appris des objectifs ramassés, qui reste faible.
4. Dans aucun cas on ne voit une **progression graduelle sur plusieurs parties**, comme chez un joueur humain.

Le problème ne semble donc pas venir des réglages. Il vient du jeu simulé ou de ce qu'on attend du modèle.

## Pourquoi l'apprentissage est trop rapide

- **Bloc M.** Au début de chaque coup, la balle est à l'arrêt : la simulation calcule x(t) = x0 + a/d·t − a/d²·(1 − e^(−d·t)), avec une vitesse initiale nulle. Le déplacement obtenu dépend donc seulement de la commande, pas de la position ni de l'histoire. C'est une fonction lisse à 2 dimensions, et un GP l'apprend avec une dizaine d'exemples.
- **Bloc D.** Sans obstacle et avec des objectifs fixes, « le plus proche d'abord » est presque optimal, et le bonus de rapprochement donne cette stratégie dès le départ.
- **Les GP en général.** Ils apprennent avec très peu d'exemples, beaucoup plus vite qu'un humain.

## Pistes pour obtenir une progression graduelle

### Piste 0 (testée) : fixer le bruit de base σ_0 des GP

Dans le schéma, σ_n = σ_0 + exp(w_r · r) − 1. Dans le code, σ_0² est le `noise_level` du `WhiteKernel`. Par défaut, il est ajusté entre 10⁻¹⁰ et 0,1 : comme la simulation est déterministe, l'optimiseur le met presque à 0, et l'agent fait alors une confiance totale à chaque observation. Le paramètre `SIGMA0_SQ` du nouveau code le fixe pour les GP des blocs D et M, en unités des sorties standardisées. `None` donne le comportement d'origine.

Niveau facile, poids du bonus 1, 15 parties, objectifs ramassés par partie (sur 3) :

| σ_0² | Graine 0 | Graine 1 |
|---|---|---|
| ajusté (code d'origine) | 0 3 3 3 3 3 3 3 3 2 3 3 3 3 3 | 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 |
| **0,1** | **0 0 1 3 3 3 3 3 3 3 3 3 3 3 2** | **1 1 0 2 2 3 3 3 3 3 3 3 3 3 3** |
| 0,5 | 0 1 3 0 3 0 3 0 3 0 3 0 3 3 3 | 1 1 0 0 0 2 2 3 0 0 3 3 3 0 3 |
| 1,0 | 0 1 0 0 0 3 2 0 3 3 3 0 3 3 3 | 1 0 1 0 0 1 3 2 2 0 3 3 3 0 0 |

- Avec **σ_0² = 0,1**, la progression s'étale sur 3 à 5 parties au lieu d'une seule, puis le score reste stable. C'est la valeur par défaut du nouveau code.
- Avec 0,5 ou 1,0, l'apprentissage est plus lent mais instable : les scores alternent entre 0 et 3. L'agent fait trop peu confiance à ce qu'il observe pour stabiliser sa stratégie.
- Ça ne demande aucun nouveau code, et ça correspond directement au terme σ_0 du schéma.
- Interprétation possible : σ_0 représente le bruit que l'agent attribue à sa perception et à sa mémoire. On pourrait l'ajuster sur les courbes d'apprentissage des joueurs.

### Piste 1 : rendre le contrôle de la balle plus dur à apprendre (bloc M)

Chez un humain, une grande partie du progrès dans Roll-a-Ball est motrice : doser l'élan, freiner, ne pas dépasser l'objectif.

- **Inertie.** La balle garde sa vitesse d'un coup à l'autre : x(t) = x0 + a/d·t + (v0 − a/d)·(1 − e^(−d·t))/d. Le déplacement dépend alors de la commande *et* de la vitesse actuelle v0. M doit apprendre une fonction de 4 variables au lieu de 2, et la vitesse actuelle doit lui être donnée comme contexte (elle est observable dans le jeu). C'est plus fidèle au vrai Roll-a-Ball (physique Unity avec `Rigidbody`).
- **Bruit moteur.** Ajouter un petit bruit gaussien à la commande appliquée, proportionnel à son intensité (bruit dépendant du signal, comme chez l'humain). La même commande ne donne plus exactement le même résultat.

Changement à faire : uniquement la méthode `step` de l'environnement, plus la vitesse ajoutée au contexte de M. Les blocs et le modèle de valeur restent identiques.

### Piste 2 : donner au bloc D quelque chose à découvrir

- **Objectifs placés au hasard à chaque partie.** D ne peut plus apprendre un trajet par cœur : V doit généraliser à partir de la position relative des objectifs, ce qui impose de les mettre dans l'état.
- **Murs intérieurs.** Un objectif derrière un mur oblige à s'éloigner avant de se rapprocher : le bonus de rapprochement induit alors en erreur, et seule la valeur V apprise permet de trouver le détour.

Changement à faire : l'environnement (collisions avec des segments), plus l'état de V pour les objectifs aléatoires.

### Piste 3 : limiter l'agent (la plus discutable)

Mémoire plus courte (wr plus grand ou `MEMORY_SIZE` plus petit), perception bruitée de la position, exploration qui diminue au fil des parties. Ces limites sont arbitraires si elles ne viennent pas de données sur les joueurs.

### Recommandation

Piste 0 (σ_0² fixé à 0,1) : déjà en place, elle étale la progression sur quelques parties sans nouveau code. Ensuite, piste 1 (inertie, plus éventuellement le bruit moteur) pour une progression plus longue et plus fidèle au jeu. Puis piste 2 si le jeu Unity a des obstacles ou des objectifs aléatoires.

Pour vérifier : on doit voir le score par partie et l'erreur de prédiction de M évoluer progressivement sur plusieurs parties, et rester nettement au-dessus du hasard.

## Questions

1. **Un apprentissage en 1 ou 2 parties suffit-il ?** Si l'analyse porte surtout sur les signaux d'agentivité *à l'intérieur* d'une partie, l'agent actuel avec bonus convient. S'il faut une progression *sur plusieurs parties*, comme chez les joueurs, ce n'est pas le cas.

2. **Dans le cas d'une progression lente, d'où doit-elle venir ?**
   - du **contrôle de la balle** (bloc M, piste 1) ;
   - de la **stratégie** (bloc D, piste 2) ;
   - des deux.
   - Faut-il fixer σ_0 (piste 0), et si oui, à partir de quoi : une valeur choisie, ou un ajustement sur les données des joueurs ?

3. **Comment le joueur contrôle-t-il la balle dans le jeu Unity ?**
   - Contrôle continu (touches maintenues) ou actions séparées ?
   - Si les actions sont séparées, la balle garde-t-elle son élan d'une action à l'autre ?
   - Qu'est-ce qui compte comme « une action » dans les données : un appui sur une touche, une fenêtre de temps fixe ?

   La réponse détermine comment modéliser l'inertie (piste 1).

4. **Le jeu Unity a-t-il des murs intérieurs, ou des objectifs placés au hasard ?** Si oui, il faut les ajouter à la simulation (piste 2).

5. **Le bonus de rapprochement est-il acceptable ?** Il représente ce que le joueur *voit*, à savoir la distance aux objectifs. Mais il est propre aux jeux où il faut atteindre une cible, et il rend la stratégie presque gratuite tant qu'il n'y a pas d'obstacle.

## Paramètres modifiables (en haut de `roll_a_ball_progress.py`)

| Paramètre | Rôle |
|---|---|
| `PROGRESS_WEIGHT` | poids du bonus de rapprochement (0 = désactivé) |
| `OBJECTIVES` | positions des objectifs |
| `ARENA` | limites de l'arène (murs) |
| `N_STEPS`, `N_EPISODES` | coups par partie, nombre de parties |
| `D_RADIUS` | distance maximale visée par le bloc D à chaque coup |
| `GAMMA`, `LR`, `WR`, `MEMORY_SIZE` | réglages de l'apprentissage de V |
| `USE_REMAINING_MASK` | ajoute à l'état les objectifs qui restent |
| `SIGMA0_SQ` | bruit de base σ_0² des GP (None = ajusté, comme dans le code d'origine) |
| `OUTPUT_NAME` | nom des fichiers de sortie (csv et html) |
