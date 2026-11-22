# Roll-a-Ball v2 : des signaux internes à la place de la récompense

Fichier : `roll_a_ball_progress_v2.py`, à placer à la racine du dépôt. Pour le lancer :

```
python roll_a_ball_progress_v2.py
```

Comme la v1, le fichier est autonome : il n'importe que le code d'origine (`src.utils`, `src.model`), qui n'est pas modifié. Il produit `data/RollABall/signals_v2.csv` et `signals_v2.html`.

**Statut : non testé.** J'ai seulement vérifié que la syntaxe est correcte. Aucun lancement n'a été fait.

## Idée

Il n'y a pas de récompense venant de l'extérieur. Ce qui guide l'agent, ce sont ses signaux internes (les signaux d'agentivité). Pour choisir où aller, le bloc D prédit, pour chaque cible possible, une **combinaison pondérée des signaux** qui suivront l'action. Il choisit ensuite la cible dont la combinaison est la plus haute.

Les poids sont **adaptatifs** : un signal qui ne varie plus perd son poids. Par exemple, si on tire deux fois la balle et que le score ne change pas, le score cesse de compter.

## Changements par rapport à la v1 (`roll_a_ball_progress.py`)

| | v1 | v2 |
|---|---|---|
| Ce qui guide le bloc D | une récompense = objectifs ramassés + `PROGRESS_WEIGHT` × rapprochement | une combinaison pondérée de 7 signaux internes |
| Ce que le GP du bloc D apprend | une valeur V(s) par TD(0) : récompense + γ·V(état suivant) | la combinaison de signaux qui suit l'action (horizon d'un coup) |
| Poids | un seul poids fixe (`PROGRESS_WEIGHT`) | un poids par signal, ajusté automatiquement selon que le signal varie encore |
| Mémoire | valeurs TD stockées une fois pour toutes | signaux **bruts** stockés ; la combinaison est recalculée avec les poids actuels à chaque mise à jour |
| Nombre d'essais | 30 parties × 6 coups = jusqu'à 180 | 10 parties × 4 coups = 40, comme les expériences d'origine (20 × 2) |
| Paramètres supprimés | — | `PROGRESS_WEIGHT`, `GAMMA`, `LR` |
| Paramètres ajoutés | — | `SIGNALS`, `RELEVANCE_WINDOW`, `PROGRESS_WINDOW` |
| Inchangé | | environnement, bloc M, corrections A1-A4, GP hétéroscédastique, `WR` = 3, mémoire de 200 avec t_max = min(t+1, 200), `SIGMA0_SQ` = 0,1 |

### Pourquoi le TD a été retiré

- Le TD propage une **récompense** vers les états précédents. Sans récompense, il n'a plus de rôle direct.
- Les poids changent au fil du temps. Il faut donc pouvoir recalculer les anciennes valeurs avec les poids actuels, ce qui est simple avec des signaux bruts mais pas avec des cibles TD.
- `goal_closure`, la distance à l'objectif, est un signal disponible à chaque coup. Il guide déjà vers les objectifs sans qu'il faille anticiper plusieurs coups.

Si on veut tenir compte des coups suivants, on peut remettre un TD sur la combinaison. Il faudra alors décider comment traiter le changement des poids.

## Calcul

Après chaque coup, on lit les 7 signaux :

| Signal | D'où il vient |
|---|---|
| `reward` | objectifs ramassés pendant le coup (le score) |
| `goal_closure` | distance de la balle à l'objectif restant le plus proche |
| `competence_progress` | évolution de `goal_closure` sur les `PROGRESS_WINDOW` derniers coups (moyenne de la moitié récente − moyenne de la moitié ancienne, comme `BlockAgency`) |
| `prediction_error` | bloc M : écart entre le déplacement prédit et le déplacement réel |
| `learning_progress` | bloc M : évolution de l'erreur de prédiction |
| `feeling_of_effort` | bloc M : changement de commande entre deux coups |
| `convergence_selected_action` | bloc M : variation du meilleur score à la fin de la recherche d'action |

Ensuite :

```
z_i         = (signal_i − moyenne_i) / écart-type_i        sur toute la mémoire
relevance_i = écart-type_i sur les RELEVANCE_WINDOW derniers coups / écart-type_i sur toute la mémoire   (borné à [0, 1])
w_i         = SIGN_i × BASE_WEIGHT_i × relevance_i
combinaison = Σ w_i × z_i
```

Un signal constant ces derniers coups a une relevance proche de 0, donc un poids proche de 0. Un signal qui varie autant qu'avant a une relevance de 1.

Le GP du bloc D apprend la relation (cible visée, objectifs restants) → combinaison. À chaque mise à jour, il est réajusté sur toute la mémoire, avec la combinaison recalculée à partir des poids actuels.

## Choix à valider

Ces points n'ont pas été précisés. J'ai pris des valeurs par défaut, toutes modifiables dans `SIGNALS` en haut du fichier.

1. **Signes des signaux.** Ils indiquent si l'agent cherche des valeurs hautes (+1) ou basses (−1) :
   - `reward` +1 et `goal_closure` −1 : sans ambiguïté.
   - `competence_progress` −1 et `learning_progress` −1 : dans le code d'origine, ils sont négatifs quand on progresse (la distance ou l'erreur baisse), donc −1 veut dire que l'agent recherche le progrès.
   - `prediction_error` −1 : l'agent préfère les actions dont il prévoit le résultat (sentiment de contrôle). Le choix inverse (+1, curiosité) est défendable.
   - `feeling_of_effort` −1 : l'effort coûte.
   - `convergence_selected_action` −1 : l'agent préfère les décisions sur lesquelles il converge vite.
2. **Poids de base** : tous à 1.
3. **Origine des signaux** : `reward`, `goal_closure` et `competence_progress` sont mesurés au niveau du jeu (objectifs réels). Les quatre autres viennent du bloc M, c'est-à-dire du contrôle moteur. Les signaux du bloc D ne sont pas utilisés, pour éviter que D se guide sur ses propres prédictions.
4. **Règle d'adaptation des poids** : le rapport des écarts-types, récents sur ensemble de la mémoire. D'autres règles sont possibles, par exemple une corrélation entre le signal et l'action choisie.
5. **`competence_progress` à travers les parties** : l'historique n'est pas remis à zéro entre deux parties, comme pour les blocs. Au début d'une partie, la distance remonte d'un coup, ce qui crée un faux « recul ».

## Paramètres (en haut du fichier)

| Paramètre | Rôle |
|---|---|
| `SIGNALS` | signe et poids de base de chaque signal (signe 0 = signal ignoré) |
| `RELEVANCE_WINDOW` | nombre de coups récents utilisés pour juger si un signal varie encore |
| `PROGRESS_WINDOW` | fenêtre de `competence_progress` |
| `N_EPISODES`, `N_STEPS` | parties, coups par partie |
| `OBJECTIVES`, `ARENA` | positions des objectifs, limites de l'arène |
| `D_RADIUS` | distance maximale visée par D à chaque coup |
| `WR`, `MEMORY_SIZE`, `SIGMA0_SQ` | oubli, taille de la mémoire, bruit de base σ_0² des GP |
| `USE_REMAINING_MASK` | ajoute les objectifs restants à l'entrée du GP de D |

## Sorties

Le CSV contient une ligne par coup, avec :
- les signaux bruts (`sig_*`) ;
- les poids au moment du coup (`w_*`) ;
- la combinaison obtenue (`value`) ;
- les objectifs ramassés ;
- les signaux d'agentivité des deux blocs, comme dans les expériences d'origine.

Le HTML trace `collected`, `value`, les signaux et les poids en fonction des essais, avec les limites des parties en fond.

## À vérifier au premier lancement

- Que l'agent ramasse plus d'objectifs qu'un agent qui joue au hasard (environ 0,07 par partie en v1).
- Que les poids évoluent comme prévu. Par exemple, `w_reward` doit rester proche de 0 tant que rien n'est ramassé.
- La durée : elle devrait être nettement plus courte qu'en v1, puisqu'il n'y a que 40 essais.
