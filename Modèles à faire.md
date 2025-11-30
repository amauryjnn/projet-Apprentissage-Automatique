# Stratégie de Modélisation et Optimisation des Hyperparamètres

Ce document recense l'ensemble des architectures de modèles de classification pertinentes pour le projet de prédiction d'approbation de prêt (`loan_status`). Il inclut les modèles déjà implémentés ainsi que les alternatives à explorer.

Pour chaque modèle, nous détaillons sa particularité technique, sa pertinence vis-à-vis du dataset (données tabulaires, mix numérique/catégoriel, déséquilibre de classe) et les hyperparamètres critiques à optimiser.

---

## 1. Modèles Linéaires (Baseline)

Ces modèles servent de référence. Ils sont interprétables et rapides, mais limités aux relations linéaires entre les variables et la cible.

### Logistic Regression (Déjà implémenté)
**Particularité :** Modèle probabiliste estimant la log-vraisemblance d'appartenance à une classe.
**Pourquoi ce modèle :** Il permet d'obtenir une baseline robuste et d'analyser l'importance des features via les coefficients (poids).

#### Fonctionnement mathématique

Le modèle réalise d'abord une **combinaison linéaire** des variables d'entrée, puis applique la **fonction sigmoïde** pour obtenir une probabilité :

$$z = b + \sum_{j=1}^{p} w_j \cdot x_j$$

$$P(y=1) = \frac{1}{1 + e^{-z}}$$

Où :
- $w_j$ : poids (coefficient) de la feature $j$
- $b$ : biais (ordonnée à l'origine)
- $x_j$ : valeur de la feature $j$
- $P(y=1)$ : probabilité prédite (entre 0 et 1)

#### La fonction de coût totale et le rôle de `C`

Le modèle minimise la fonction suivante :

$$\text{Coût Total} = \underbrace{\frac{1}{n} \sum_{i=1}^{n} \text{Log-Loss}_i}_{\text{Erreur de prédiction}} + \underbrace{\frac{1}{2C} \sum_{j=1}^{p} w_j^2}_{\text{Pénalité de régularisation (L2)}}$$

**Où :**
- **Log-Loss** : $-[y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$ (pénalise les mauvaises prédictions)
- **Pénalité L2** : Force les poids à rester petits pour éviter le surapprentissage

**Rôle du paramètre `C` (inverse de la force de régularisation) :**

| Valeur de C | $\frac{1}{C}$ | Effet |
|---|---|---|
| **C petit** (0.01) | **Très grand** | Pénalité forte → modèle simple et rigide → **sous-apprentissage possible** |
| **C = 1.0** | 1 | Équilibre optimal (défaut) |
| **C grand** (100) | **Très petit** | Pénalité faible → modèle complexe → **surapprentissage possible** |

**Hyperparamètres à optimiser :**
*   `C` (Inverse de la force de régularisation) :
    *   *Justification :* Un `C` faible augmente la régularisation (évite le surapprentissage), un `C` fort laisse le modèle s'ajuster plus étroitement aux données.
    *   *Plage recommandée :* Échelle logarithmique `[0.01, 0.1, 1, 10, 100]`.
*   `class_weight` :
    *   *Justification :* Essentiel pour gérer le déséquilibre des classes (refus vs accord) sans nécessairement passer par un undersampling destructif.
    *   *Valeur :* `'balanced'`.
*   `penalty` :
    *   *Justification :* `l1` (Lasso) peut effectuer une sélection de features en annulant certains coefficients, `l2` (Ridge) réduit l'amplitude des coefficients.


---

### Différences fondamentales

#### **BAGGING** = **B**ootstrap **Agg**regat**ing**

**Principe :**
- **Bootstrap** : Rééchantillonnage **avec remise** (on peut tirer la même observation plusieurs fois)
- **Aggregating** : Agrégation des résultats (vote simple ou moyenne)

**Construction :** Les modèles sont entraînés **en parallèle** et indépendamment les uns des autres.

**Stratégie :** Chaque modèle reçoit un sous-ensemble aléatoire différent des données. Aucun modèle n'apprend des erreurs du précédent.

**Combinaison finale :** Vote démocratique (poids égaux) ou moyenne arithmétique.

**Réduit :** La **variance** (surapprentissage)

**Exemples :** Random Forest, ExtraTreesClassifier

---

#### **BOOSTING** = **Amplification progressive**

**Principe :**
- Vient de "boost" (amplifier, renforcer)
- Les modèles s'améliorent **progressivement** en corrigeant les erreurs des précédents

**Construction :** Les modèles sont entraînés **séquentiellement** et dépendent les uns des autres.

**Stratégie :** 
1. Entraîner un 1er modèle sur les données
2. Identifier ses erreurs
3. Entraîner un 2e modèle en mettant plus de poids sur ces erreurs
4. Répéter jusqu'à N modèles

**Combinaison finale :** Somme pondérée (poids adapté à la performance de chaque modèle)

**Réduit :** Le **biais** (sous-apprentissage)

**Exemples :** XGBoost, LightGBM, AdaBoost

---

### Tableau comparatif : Bagging vs Boosting

| Aspect | **Bagging** | **Boosting** |
|--------|-----------|-----------|
| **Construction** | Parallèle (indépendants) | Séquentielle (dépendants) |
| **Données** | Bootstrap : sous-ensembles aléatoires avec remise | Tous les données, mais pondération adaptative |
| **Focus du modèle suivant** | Aucun (chaque modèle est identique) | **Sur les erreurs du précédent** |
| **Combinaison finale** | Vote/moyenne (poids égaux) | Combinaison pondérée (poids adapté) |
| **Réduit** | ↓ **Variance** (surapprentissage) | ↓ **Biais** (sous-apprentissage) |
| **Vitesse d'entraînement** | Rapide (parallélisable) | Lent (séquentiel) |
| **Surapprentissage** | Basse | Moyenne-haute (risque d'overfitting) |
| **Cas d'usage** | Modèles instables, beaucoup de données | Modèles faibles, besoin de précision max |

---

### Visualisation conceptuelle

#### Bagging (Parallèle)

```
Données originales
       ↓
   ┌───┴───┬───────┬───────┐
   ↓       ↓       ↓       ↓
[Arbre1] [Arbre2] [Arbre3] [Arbre4]  ← Entraînés indépendamment
   ↓       ↓       ↓       ↓
   └───┬───┴───┬───┴───┬───┘
       ↓       ↓       ↓
    Vote/Moyenne
       ↓
    Prédiction finale
```

**Exemple Random Forest :**
- Crée 100 sous-ensembles bootstrap de tes données
- Entraîne 100 arbres **en parallèle** (chacun sur un sous-ensemble différent)
- Combine les votes : si 60 arbres disent "approuvé" et 40 disent "rejeté" → **"approuvé"**

---

#### Boosting (Séquentiel)

```
Données originales
       ↓
   [Modèle 1] ← Entraîné sur les données
       ↓
    Erreurs ?
       ↓
   [Modèle 2] ← Entraîné en mettant plus de poids sur les erreurs du Modèle 1
       ↓
    Encore des erreurs ?
       ↓
   [Modèle 3] ← Entraîné en mettant plus de poids sur les erreurs du Modèle 2
       ↓
    ... (répéter N fois)
       ↓
   Combinaison pondérée de tous les modèles
       ↓
    Prédiction finale
```

**Exemple XGBoost :**
- Crée un 1er arbre qui fait des erreurs
- Crée un 2e arbre qui se concentre sur corriger ces erreurs
- Crée un 3e arbre qui corrige les erreurs restantes
- Les prédictions finales = somme pondérée des 3 arbres

---



## 2. Méthodes d'Ensemble - Bagging (Parallélisation)

Ces méthodes construisent plusieurs modèles indépendants en parallèle et font la moyenne de leurs prédictions pour réduire la variance (surapprentissage).

### Random Forest Classifier (Déjà implémenté)
**Particularité :** Agrégation de multiples arbres de décision entraînés sur des sous-ensembles aléatoires de données (bootstrap) et de features.
**Pourquoi ce modèle :** Très robuste, gère naturellement les interactions non-linéaires et nécessite peu de prétraitement (bien que le scaling soit déjà fait).

#### Deux méthodes de calcul : Gini vs Entropy

Le Random Forest utilise deux critères différents pour sélectionner les meilleurs points de division (splits) à chaque nœud de l'arbre :

**1. Gini Impurity (par défaut)**

$$\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$$

- $C$ : nombre de classes (ici 2 : refusé/accordé)
- $p_i$ : proportion de la classe $i$ dans le nœud

*Interprétation :* Mesure la "pureté" du nœud. Une valeur de 0 signifie que le nœud contient uniquement une classe (pur), une valeur proche de 0.5 signifie un mélange équilibré des deux classes (impur).

**2. Entropy (Information Gain)**

$$\text{Entropy} = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

*Interprétation :* Mesure l'information (ou l'incertitude) contenue dans le nœud. Une entropie élevée signifie une grande incertitude, une entropie faible signifie que le nœud est dominé par une classe.

**Comparaison pratique :**

| Aspect | Gini | Entropy |
|--------|------|---------|
| **Complexité computationnelle** | Plus rapide (simples calculs arithmétiques) | Légèrement plus lent (calculs logarithmiques) |
| **Résultats empiriques** | Généralement similaires | Généralement similaires |
| **Défaut dans scikit-learn** | ✓ Oui | Non (option) |
| **Cas d'usage** | Recommandé pour la plupart des cas | Utilisé rarement ; principalement pour comparaison |

**Comment choisir :**

Dans la pratique, il y a peu de différence entre les deux méthodes. Le choix du critère a généralement un impact négligeable comparé à d'autres hyperparamètres comme `n_estimators` ou `max_depth`. 

- Utiliser **Gini** (défaut) pour une exécution plus rapide.
- Utiliser **Entropy** si vous avez une raison théorique spécifique ou souhaitez exploiter des concepts de théorie de l'information.


**Hyperparamètres à optimiser :**
*   `n_estimators` :
    *   *Justification :* Nombre d'arbres. Plus il est élevé, plus le modèle est stable, mais plus le coût computationnel augmente.
    *   *Plage recommandée :* `[100, 200, 500]`.
*   `max_depth` :
    *   *Justification :* Contrôle la complexité de chaque arbre. Une profondeur trop grande mène au surapprentissage.
    *   *Plage recommandée :* `[10, 20, None]`.
*   `min_samples_split` / `min_samples_leaf` :
    *   *Justification :* Impose un nombre minimum d'échantillons pour créer une division. Augmenter ces valeurs lisse le modèle.
    *   *Plage recommandée :* `[2, 5, 10]`.

### ExtraTreesClassifier (À faire)
**Particularité :** Similaire au Random Forest structurellement, mais avec une différence fondamentale dans la sélection des seuils de coupure (split thresholds) aux nœuds de l'arbre.

**Pourquoi ce modèle :** Cette différence algorithme introduit plus de stochasticité (aléatoire) dans le processus d'apprentissage, ce qui réduit souvent la variance davantage que le Random Forest et accélère significativement l'entraînement.

#### Différence clé : Optimisation des splits

**Random Forest (Optimisation gourmande - Greedy)**

À chaque nœud, le Random Forest **recherche le seuil optimal** minimisant une métrique d'impureté (Gini ou Entropy) :

$$\text{Best split} = \arg\min_{j, s} \left[ \text{Impureté}_{\text{gauche}}(j, s) + \text{Impureté}_{\text{droit}}(j, s) \right]$$

**Processus :**
1. Pour chaque feature $j$ du nœud
2. Tester tous les seuils $s$ possibles (ou un sous-ensemble intelligent)
3. Évaluer chaque split et choisir le meilleur
4. **Coût computationnel élevé** : $O(\text{nb features} \times \text{nb seuils possibles})$

**ExtraTreesClassifier (Seuils aléatoires)**

ExtraTrees choisit à la place des seuils **aléatoirement** pour chaque feature et garde le meilleur parmi ces splits aléatoires :

$$\text{Best split} = \arg\min_{j \in \text{random features}, s \in \text{random thresholds}} \left[ \text{Impureté}_{\text{gauche}}(j, s) + \text{Impureté}_{\text{droit}}(j, s) \right]$$

**Processus :**
1. Sélectionner aléatoirement un sous-ensemble de features
2. Pour chaque feature, générer des seuils **aléatoires** (au lieu de les optimiser)
3. Évaluer uniquement ces splits aléatoires
4. Choisir le meilleur parmi ceux-ci
5. **Coût computationnel réduit** : $O(\text{nb features aléatoires} \times \text{nb seuils aléatoires fixes})$

#### Tableau comparatif : Random Forest vs ExtraTreesClassifier

| Aspect | Random Forest | ExtraTreesClassifier |
|--------|---------------|----------------------|
| **Sélection des splits** | Optimale (recherche gourmande) | Aléatoire |
| **Complexité computationnelle** | Élevée (O(n log n) par arbre) | Faible (O(n)) |
| **Variance du modèle** | Moyenne | Faible (aléatoire réduit la variance) |
| **Biais du modèle** | Bas | Légèrement plus haut |
| **Temps d'entraînement** | Lent | Rapide |
| **Parallelisation** | Possible mais moins efficace | Très efficace (moins de synchronisation) |
| **Sensibilité au surapprentissage** | Moyennement sensible | Moins sensible (l'aléatoire régularise) |
| **Performance typique** | Très bonne | Souvent comparable ou légèrement meilleure |
| **Besoin de tuning** | Plus critique | Moins critique grâce à la régularisation implicite |

#### Intuition graphique

Imaginons une feature continue qui sépare les données :

**Random Forest :** Teste le seuil à 5.2, 5.3, 5.5, ..., 8.9 et choisit 6.8 comme optimal.

**ExtraTreesClassifier :** Génère 3 seuils aléatoires (disons 4.2, 6.9, 7.3) et choisit le meilleur parmi ces trois (par exemple 6.9).

ExtraTrees ne trouve pas le "parfait" 6.8, mais son approche plus rapide et la diversité des arbres générés compensent cette "imprécision" locale.

#### Quand choisir ExtraTreesClassifier ?

✓ **Utiliser ExtraTreesClassifier si :**
- Vous avez besoin d'une **exécution rapide** (données volumineuses)
- Vous cherchez à **réduire le surapprentissage** naturellement
- Les **ressources computationnelles sont limitées**
- Vous tolérez une performance comparable ou légèrement inférieure en échange d'une grande vitesse

✗ **Utiliser Random Forest si :**
- La **précision est critique** et vous avez le budget computationnel
- Les données sont **petites à moyennes** (< 100k lignes)
- Vous avez besoin d'une **stabilité maximale**

**Hyperparamètres à optimiser :**
*   Identiques au Random Forest (`n_estimators`, `max_depth`, `min_samples_split`).
*   *Stratégie spécifique :* On peut souvent se permettre des arbres légèrement plus profonds qu'en Random Forest (ex: `max_depth=[50, 70, 100]`) car le caractère aléatoire des splits protège partiellement du surapprentissage, agissant comme une **régularisation implicite**.
*   *Recommandation :* Généralement, augmenter `n_estimators` pour ExtraTreesClassifier (ex: 500-1000) pour compenser la perte de précision individuelle des arbres.


---

## 3. Méthodes d'Ensemble - Boosting (Séquentiel)

Ces méthodes construisent les modèles séquentiellement, chaque nouveau modèle tentant de corriger les erreurs des précédents. C'est l'état de l'art pour les données tabulaires.

### XGBoost & LightGBM (Déjà implémentés)
**Particularité :** Algorithmes de Gradient Boosting optimisés pour la vitesse et la performance. LightGBM utilise une méthode de binning (histogrammes) et une croissance par feuille (leaf-wise), tandis que XGBoost utilise une croissance par niveau (level-wise).
**Pourquoi ce modèle :** Ils offrent généralement les meilleurs scores F1 sur ce type de données.

**Hyperparamètres à optimiser :**
*   `learning_rate` (eta) :
    *   *Justification :* Rétrécit la contribution de chaque arbre. Un taux faible nécessite plus d'arbres mais généralise mieux.
    *   *Plage recommandée :* `[0.01, 0.05, 0.1]`.
*   `num_leaves` (LightGBM) / `max_depth` (XGBoost) :
    *   *Justification :* Contrôle principal de la complexité du modèle.
*   `subsample` & `colsample_bytree` :
    *   *Justification :* Fraction des données et des colonnes utilisées par arbre. Prévient le surapprentissage.
    *   *Plage recommandée :* `[0.6, 0.8, 1.0]`.
*   `scale_pos_weight` (XGBoost) / `is_unbalance` (LightGBM) :
    *   *Justification :* Gestion du déséquilibre de classe.

### HistGradientBoostingClassifier (À faire)
**Particularité :** Implémentation native de Scikit-Learn inspirée de LightGBM. Elle discretise les données continues en histogrammes (bins).
**Pourquoi ce modèle :** Beaucoup plus rapide que le `GradientBoostingClassifier` classique pour n > 10 000 lignes. Il gère nativement les valeurs manquantes (NaN), ce qui peut simplifier le pipeline de pré-traitement.

**Hyperparamètres à optimiser :**
*   `learning_rate` : Comme pour XGBoost.
*   `max_iter` : Équivalent de `n_estimators`.
*   `max_leaf_nodes` : Remplace `max_depth` pour contrôler la complexité.
*   `l2_regularization` : Pour contraindre les poids.

### AdaBoostClassifier (À faire)
**Particularité :** Au lieu de fitter sur les résidus (comme le Gradient Boosting), AdaBoost augmente le poids des observations mal classées à chaque itération.
**Pourquoi ce modèle :** Approche différente du boosting. Souvent moins performant que XGBoost sur des données complexes, mais utile pour diversifier un ensemble de modèles (Stacking).

**Hyperparamètres à optimiser :**
*   `n_estimators` : Nombre de modèles faibles itératifs.
*   `learning_rate` : Contribution de chaque modèle.
*   `base_estimator` : Par défaut un arbre de décision très simple (stump, profondeur=1). On peut essayer d'augmenter légèrement la profondeur.

---

## 4. Méthodes basées sur les Voisins et Vecteurs

Ces modèles capturent des relations basées sur la distance géométrique dans l'espace des features.

### KNeighborsClassifier (KNN) (À faire)
**Particularité :** Algorithme non-paramétrique qui classe une observation selon la classe majoritaire de ses `k` voisins les plus proches.
**Pourquoi ce modèle :** Repose sur l'hypothèse que des profils clients similaires ont des comportements de remboursement similaires.

**Hyperparamètres à optimiser :**
*   `n_neighbors` (k) :
    *   *Justification :* Un k petit est sensible au bruit (surapprentissage), un k grand lisse trop les frontières.
    *   *Plage recommandée :* `[5, 15, 30, 50]`.
*   `weights` :
    *   *Justification :* `'uniform'` (vote égal) ou `'distance'` (les voisins proches ont plus de poids). `'distance'` est souvent préférable en haute dimension.
*   `metric` :
    *   *Justification :* `euclidean` ou `manhattan`.

### LinearSVC (À faire)
**Particularité :** Cherche l'hyperplan qui maximise la marge entre les deux classes. Version linéaire du SVM.
**Pourquoi ce modèle :** Le SVC avec noyau (RBF) a une complexité quadratique $O(n^2)$, ce qui est trop lent pour 45 000 lignes. `LinearSVC` est beaucoup plus rapide ($O(n)$) et performant si les données sont linéairement séparables dans l'espace transformé.

**Hyperparamètres à optimiser :**
*   `C` :
    *   *Justification :* Contrôle la dureté de la marge. Un C faible tolère plus d'erreurs de classification (marge souple).
*   `dual` :
    *   *Justification :* Mettre à `False` car le nombre d'échantillons > nombre de features.

---

## 5. Réseaux de Neurones

Capables d'apprendre des représentations complexes et non-linéaires des données.

### Deep MLP (PyTorch) (Déjà implémenté)
**Particularité :** Architecture personnalisée avec Batch Normalization, Dropout et connexions résiduelles.
**Pourquoi ce modèle :** Flexibilité totale sur l'architecture et l'optimisation.

**Hyperparamètres à optimiser :**
*   `Learning Rate` & `Scheduler` : Critique pour la convergence.
*   `Dropout rate` : Pour la régularisation.
*   `Architecture` : Nombre de couches et neurones par couche.

### MLPClassifier (Scikit-Learn) (À faire)
**Particularité :** Implémentation standard du Perceptron Multicouche dans Scikit-Learn.
**Pourquoi ce modèle :** Moins flexible que PyTorch mais intégration immédiate dans les Pipelines Scikit-Learn et GridSearchCV. Utile pour valider rapidement si une approche neuronale est pertinente sans coder une boucle d'entraînement complexe.

**Hyperparamètres à optimiser :**
*   `hidden_layer_sizes` :
    *   *Justification :* Structure du réseau. Ex: `(100,)` ou `(100, 50)`.
*   `alpha` :
    *   *Justification :* Régularisation L2 (poids decay).
    *   *Plage recommandée :* `[0.0001, 0.001, 0.01]


### TabNet (Google Research) (À faire)
**Particularité :** Architecture de Deep Learning utilisant un mécanisme d'attention séquentielle (Sequential Attention) pour sélectionner les features pertinentes à chaque étape de décision. Elle imite la logique hiérarchique des arbres de décision tout en conservant les avantages de l'apprentissage de bout en bout (end-to-end learning).
**Pourquoi ce modèle :** C'est actuellement l'une des rares architectures neuronales capable de rivaliser avec les performances des Gradient Boosting Decision Trees (GBDT) sur des données tabulaires. Elle offre également une interprétabilité native grâce aux masques d'attention (feature selection masks).

**Hyperparamètres à optimiser :**
*   `n_d` et `n_a` (Dimensions) :
    *   *Justification :* Taille des représentations pour la prédiction (`n_d`) et pour l'attention (`n_a`). Généralement, on fixe `n_d = n_a`.
    *   *Plage recommandée :* `[8, 16, 32, 64]`.
*   `n_steps` :
    *   *Justification :* Nombre d'étapes de décision séquentielles (nombre de fois où le modèle "regarde" les données).
    *   *Plage recommandée :* `[3, 5, 7]`.
*   `gamma` :
    *   *Justification :* Coefficient de relaxation qui contrôle la réutilisation des features. Une valeur proche de 1 force le modèle à utiliser des features différentes à chaque étape.
    *   *Plage recommandée :* `[1.0, 1.2, 1.5]`.
*   `lambda_sparse` :
    *   *Justification :*### TabNet (Google Research) (À faire)
**Particularité :** Architecture de Deep Learning utilisant un mécanisme d'attention séquentielle (Sequential Attention) pour sélectionner les features pertinentes à chaque étape de décision. Elle imite la logique hiérarchique des arbres de décision tout en conservant les avantages de l'apprentissage de bout en bout (end-to-end learning).
**Pourquoi ce modèle :** C'est actuellement l'une des rares architectures neuronales capable de rivaliser avec les performances des Gradient Boosting Decision Trees (GBDT) sur des données tabulaires. Elle offre également une interprétabilité native grâce aux masques d'attention (feature selection masks).

**Hyperparamètres à optimiser :**
*   `n_d` et `n_a` (Dimensions) :
    *   *Justification :* Taille des représentations pour la prédiction (`n_d`) et pour l'attention (`n_a`). Généralement, on fixe `n_d = n_a`.
    *   *Plage recommandée :* `[8, 16, 32, 64]`.
*   `n_steps` :
    *   *Justification :* Nombre d'étapes de décision séquentielles (nombre de fois où le modèle "regarde" les données).
    *   *Plage recommandée :* `[3, 5, 7]`.
*   `gamma` :
    *   *Justification :* Coefficient de relaxation qui contrôle la réutilisation des features. Une valeur proche de 1 force le modèle à utiliser des features différentes à chaque étape.
    *   *Plage recommandée :* `[1.0, 1.2, 1.5]`.
*   `lambda_sparse` :
    *   *Justification :* Pénalité de parcimonie (sparsity) sur les masques d'attention. Plus elle est élevée, plus le modèle essaie de se concentrer sur un petit nombre de features.

---

## 6. Méthodes d'Ensemble - Hybridation (Méta-modèles)

Ces techniques combinent les prédictions de plusieurs modèles hétérogènes (ex: un modèle linéaire + un arbre + un réseau de neurones) pour obtenir une performance supérieure à celle du meilleur modèle individuel.

### VotingClassifier (Scikit-Learn) (À faire)
**Particularité :** Agrège les résultats de plusieurs classifieurs (estimators) entraînés séparément.
**Pourquoi ce modèle :** Permet de lisser les erreurs spécifiques à un type d'algorithme. Si la Régression Logistique rate une relation non-linéaire mais que le Random Forest la capte, le vote permet de corriger le tir. C'est souvent l'étape ultime pour gagner les derniers points de pourcentage de précision.

**Hyperparamètres à optimiser :**
*   `voting` :
    *   *Justification :*
        *   `'hard'` : Vote majoritaire simple sur les classes prédites (0 ou 1). La classe qui a le plus de voix l'emporte.
        *   `'soft'` : Moyenne pondérée des probabilités prédites par chaque modèle. Généralement plus performant car il prend en compte la "confiance" des modèles, mais nécessite que les modèles sortent des probabilités calibrées (`predict_proba`).
*   `weights` :
    *   *Justification :* Permet de donner plus d'importance aux modèles les plus performants (ex: donner un poids de 3 à XGBoost, 1 à la Régression Logistique et 2 au Random Forest).