# Questions & Réponses - Préparation Soutenance Projet "Loan Approval"

Ce document recense **plus de 100 questions** techniques, méthodologiques et de mise en situation qu'un jury expert pourrait vous poser. Elles couvrent l'intégralité de votre pipeline, du nettoyage des données au déploiement théorique.

---

## I. Compréhension des Données (EDA)

### 1. Analyse Univariée & Distributions
**Q1 :** En observant la distribution de `person_age`, vous avez identifié une valeur maximale de 144 ans. Quelle a été votre décision et pourquoi ?
**R1 :** C'est une valeur aberrante (outlier), probablement une erreur de saisie. Nous l'avons supprimée car elle risque de fausser la normalisation (StandardScaler) et d'impacter négativement les modèles sensibles aux distances.

**Q2 :** La variable cible `loan_status` est-elle équilibrée ? Quelles sont les proportions initiales ?
**R2 :** Non, elle est déséquilibrée (environ 78% de prêts accordés vs 22% de refus). Cela impose d'utiliser des métriques adaptées (F1-score) et des techniques de rééchantillonnage.

**Q3 :** Vous avez utilisé `scatter_matrix`. Que cherchiez-vous à visualiser entre `person_income` et `loan_amnt` ?
**R3 :** Nous cherchions une corrélation positive (plus on gagne, plus on emprunte) et à détecter des anomalies (ex: faible revenu mais emprunt massif).

**Q4 :** Quelle est la distribution de la variable `loan_intent` ? Y a-t-il une catégorie sous-représentée ?
**R4 :** Les catégories sont relativement bien réparties (Education, Medical, Venture, etc.), mais il faut vérifier si certaines catégories rares (ex: "Home Improvement") ne posent pas problème lors du split train/test (risque d'absence dans le test set).

**Q5 :** Avez-vous observé une distribution normale (gaussienne) sur `person_income` ?
**R5 :** Non, les revenus suivent généralement une distribution log-normale (asymétrie à droite, "right-skewed"). La plupart des gens ont des revenus moyens, et une petite minorité a des revenus très élevés.

**Q6 :** Pourquoi avoir converti `person_age` en entier (`int`) dès le début ?
**R6 :** Pour la cohérence sémantique (un âge est un entier) et pour économiser un peu de mémoire, même si l'impact est minime ici.

**Q7 :** Avez-vous détecté des valeurs négatives dans des colonnes où cela serait impossible (ex: `person_emp_exp`) ?
**R7 :** C'est un point de vérification crucial. Si `person_emp_exp` (expérience pro) était négative, ce serait une erreur de saisie à corriger (valeur absolue ou suppression).

**Q8 :** Que signifie une valeur de 0 pour `credit_score` ? Est-ce une donnée manquante ou un score réel ?
**R8 :** Dans ce dataset synthétique, cela peut être un score réel très bas ou une valeur par défaut. Il faudrait vérifier la documentation. Si c'est une valeur manquante déguisée, il faut l'imputer.

**Q9 :** Comment se répartit `person_home_ownership` ?
**R9 :** Les modalités principales sont RENT, OWN, MORTGAGE. La catégorie OTHER est très minoritaire.

**Q10 :** Avez-vous regardé la distribution conjointe de `loan_grade` et `loan_int_rate` ?
**R10 :** Oui, on s'attend à ce que les grades "risqués" (D, E, F) aient des taux d'intérêt plus élevés. C'est une vérification de cohérence métier.

### 2. Corrélations
**Q11 :** Pourquoi la matrice de corrélation affiche-t-elle une forte corrélation entre `loan_percent_income`, `loan_amnt` et `person_income` ?
**R11 :** C'est une corrélation structurelle (multicolinéarité) : `loan_percent_income` est mathématiquement dérivé des deux autres.

**Q12 :** Quel type de corrélation mesure le coefficient de Pearson ?
**R12 :** Uniquement la corrélation **linéaire**. Une relation en forme de U (quadratique) donnerait un Pearson proche de 0.

**Q13 :** Pourquoi utiliser le test du Chi² pour `person_education` vs `loan_status` ?
**R13 :** Parce que ce sont deux variables catégorielles. Pearson ne s'applique pas ici. Le Chi² teste l'indépendance entre les deux variables.

**Q14 :** Qu'est-ce que le V de Cramér et pourquoi l'avoir calculé ?
**R14 :** Le Chi² nous dit *si* deux variables sont liées (significativité), le V de Cramér nous dit *à quel point* elles sont liées (intensité, entre 0 et 1).

**Q15 :** Avez-vous trouvé une variable numérique fortement corrélée négativement avec `loan_status` ?
**R15 :** Potentiellement `loan_percent_income` ou `loan_int_rate` (plus le taux/ratio est élevé, moins on a de chance d'être approuvé, ou inversement selon le codage de la target).

---

## II. Nettoyage et Pré-traitement (Preprocessing)

### 3. Gestion des valeurs manquantes
**Q16 :** Vous avez des valeurs manquantes dans `person_emp_length`. Pourquoi avoir choisi la médiane plutôt que la moyenne ?
**R16 :** La médiane est robuste aux valeurs extrêmes (outliers). Si un CEO gagne 10M€, la moyenne des revenus explose, mais la médiane reste stable.

**Q17 :** Pourquoi ne pas avoir utilisé un `KNNImputer` ?
**R17 :** C'est une méthode plus précise mais beaucoup plus coûteuse en temps de calcul ($O(n^2)$). Pour une première approche, l'imputation simple (médiane/mode) suffit souvent.

**Q18 :** Comment avez-vous géré les valeurs manquantes pour les variables catégorielles ?
**R18 :** En créant une catégorie explicite "Missing" ou en imputant par le mode (la valeur la plus fréquente).

**Q19 :** Si vous aviez 50% de valeurs manquantes dans une colonne, que feriez-vous ?
**R19 :** Je supprimerais probablement la colonne, sauf si le fait que la donnée soit manquante est une information en soi (ex: pas de score de crédit = jamais emprunté).

### 4. Encodage et Scaling
**Q20 :** Pourquoi utiliser `OneHotEncoder` pour `loan_intent` plutôt que `LabelEncoder` ?
**R20 :** `LabelEncoder` introduit un ordre arbitraire (Education < Medical < Venture) qui fausserait les modèles linéaires et neuronaux. `OneHotEncoder` traite chaque catégorie équitablement.

**Q21 :** Quel est l'inconvénient du `OneHotEncoder` si une variable a 1000 catégories ?
**R21 :** "The Curse of Dimensionality". Cela crée 1000 colonnes supplémentaires, rendant le dataset énorme et creux (sparse), ce qui ralentit l'entraînement et augmente le risque d'overfitting.

**Q22 :** Dans ce cas (1000 catégories), que feriez-vous ?
**R22 :** J'utiliserais du `Target Encoding` (remplacer la catégorie par la moyenne de la target pour cette catégorie) ou je regrouperais les catégories rares dans "Other".

**Q23 :** Pourquoi le `StandardScaler` est-il appliqué *après* le split train/test ?
**R23 :** Pour éviter le **Data Leakage**. Les statistiques (moyenne, écart-type) doivent être calculées uniquement sur le train set, puis appliquées au test set. Sinon, le modèle "connaît" la distribution du test set avant de l'avoir vu.

**Q24 :** Le `StandardScaler` est-il utile pour XGBoost ?
**R24 :** Non, les arbres de décision ne se soucient pas de l'échelle des variables. Mais il est indispensable pour la Régression Logistique et le Réseau de Neurones.

**Q25 :** Pourquoi avoir utilisé `handle_unknown='ignore'` dans le OneHotEncoder ?
**R25 :** Pour que le modèle ne plante pas en production s'il rencontre une nouvelle catégorie jamais vue dans le train set (ex: un nouveau type de prêt).

### 5. Gestion du déséquilibre
**Q26 :** Vous avez utilisé `RandomUnderSampler`. Quel est le risque majeur ?
**R26 :** La perte d'information massive. On jette des données potentiellement utiles de la classe majoritaire.

**Q27 :** Pourquoi ne pas avoir utilisé SMOTE (Sur-échantillonnage) ?
**R27 :** SMOTE crée des données synthétiques qui peuvent introduire du bruit ou des frontières de décision floues si les classes se chevauchent. De plus, cela augmente la taille du dataset d'entraînement, ralentissant les calculs.

**Q28 :** Si vous n'aviez pas fait d'undersampling, comment auriez-vous géré le déséquilibre ?
**R28 :** En utilisant le paramètre `class_weight='balanced'` dans les modèles, qui pénalise davantage les erreurs sur la classe minoritaire.

**Q29 :** L'undersampling doit-il être appliqué sur le Test Set ?
**R29 :** **JAMAIS !** Le Test Set doit refléter la réalité (déséquilibrée) de la production. On ne rééquilibre que le Train Set.

---

## III. Modélisation Classique (Logistic Regression, RF)

### 6. Régression Logistique
**Q30 :** Pourquoi commencer par une Régression Logistique ?
**R30 :** C'est la "Baseline". Si un modèle complexe ne bat pas la régression logistique, il ne sert à rien. Elle est aussi très interprétable (poids des coefficients).

**Q31 :** Que signifie un coefficient positif pour la variable `income` ?
**R31 :** Cela signifie que plus le revenu augmente, plus la probabilité d'appartenir à la classe 1 (Prêt Accordé) augmente (toutes choses égales par ailleurs).

**Q32 :** À quoi sert le paramètre `C` ?
**R32 :** C'est l'inverse de la régularisation. Petit C = Forte régularisation (modèle simple, risque de sous-apprentissage). Grand C = Faible régularisation (modèle complexe, risque de sur-apprentissage).

**Q33 :** Quelle est la différence entre pénalité L1 et L2 ?
**R33 :** L1 (Lasso) peut mettre certains coefficients à zéro (sélection de variables). L2 (Ridge) réduit tous les coefficients vers zéro sans les annuler complètement.

**Q34 :** Pourquoi `max_iter=1000` ?
**R34 :** L'algorithme d'optimisation (ex: lbfgs) peut avoir besoin de plus d'itérations pour converger si les données sont complexes ou mal mises à l'échelle.

### 7. Random Forest
**Q35 :** Expliquez le principe du Bagging.
**R35 :** Bootstrap Aggregating. On crée plusieurs datasets en tirant des échantillons avec remise, on entraîne un modèle sur chaque, et on moyenne les prédictions. Cela réduit la variance.

**Q36 :** Pourquoi le Random Forest ne risque-t-il pas trop l'overfitting par rapport à un arbre unique ?
**R36 :** Parce qu'il moyenne les erreurs de nombreux arbres décorrélés. L'erreur de généralisation converge vers une limite quand le nombre d'arbres augmente.

**Q37 :** À quoi sert `min_samples_split` ?
**R37 :** C'est le nombre minimum d'échantillons requis pour diviser un nœud. Plus il est élevé, plus l'arbre est simple (moins profond), ce qui régularise le modèle.

**Q38 :** Quelle est la différence entre `RandomForestClassifier` et `ExtraTreesClassifier` ?
**R38 :** Dans ExtraTrees, les seuils de coupure sont choisis aléatoirement (au lieu de chercher le meilleur seuil localement). Cela va plus vite et réduit encore plus la variance.

**Q39 :** Comment le Random Forest gère-t-il les features non pertinentes ?
**R39 :** Il les ignore naturellement car elles ne seront jamais choisies comme meilleur critère de division (split) dans les nœuds.

**Q40 :** Le Random Forest peut-il extrapoler des valeurs hors du domaine d'entraînement (ex: un revenu 2x supérieur au max connu) ?
**R40 :** Non. Contrairement à une régression linéaire, un arbre ne peut pas prédire une valeur supérieure à la moyenne de la feuille la plus haute.

---

## IV. Boosting (XGBoost, LightGBM)

### 8. Principes Généraux
**Q41 :** Quelle est la différence fondamentale entre Bagging (RF) et Boosting (XGB/LGBM) ?
**R41 :** Le Bagging entraîne les modèles en **parallèle** (indépendants). Le Boosting les entraîne en **série** (séquentiels), chaque modèle corrigeant les erreurs du précédent.

**Q42 :** Pourquoi le Boosting est-il souvent meilleur sur données tabulaires ?
**R42 :** Parce qu'il se concentre spécifiquement sur les cas difficiles (les erreurs), ce qui permet de modéliser des frontières de décision très complexes.

**Q43 :** Quel est le risque principal du Boosting ?
**R43 :** Il est très sensible au bruit et aux outliers, car il va essayer de les corriger à tout prix, menant au surapprentissage.

### 9. LightGBM & XGBoost
**Q44 :** Pourquoi LightGBM est-il plus rapide que XGBoost classique ?
**R44 :** Il utilise le "Histogram-based algorithm" (discrétisation des valeurs continues en bins) et une croissance de l'arbre "Leaf-wise" (par feuille) plutôt que "Level-wise" (par niveau).

**Q45 :** Vous avez eu un problème de lenteur avec LightGBM. Expliquez.
**R45 :** C'était un "Deadlock" dû à la parallélisation imbriquée. `GridSearchCV` utilisait tous les cœurs (`n_jobs=-1`) ET chaque modèle LightGBM essayait aussi d'utiliser tous les cœurs. Il fallait mettre `n_jobs=1` dans le modèle.

**Q46 :** À quoi sert le `learning_rate` (eta) ?
**R46 :** Il pondère la contribution de chaque nouvel arbre. Un petit LR nécessite plus d'arbres mais généralise mieux. C'est l'hyperparamètre le plus important à tuner avec `n_estimators`.

**Q47 :** Que fait le paramètre `subsample` ?
**R47 :** Il indique la fraction des données à utiliser pour entraîner chaque arbre (Stochastic Gradient Boosting). Cela ajoute de l'aléatoire et prévient l'overfitting.

**Q48 :** Comment XGBoost gère-t-il les valeurs manquantes ?
**R48 :** Il apprend automatiquement une direction par défaut ("default direction") pour les valeurs manquantes lors de la construction de l'arbre.

**Q49 :** Pourquoi utiliser `eval_metric='logloss'` ?
**R49 :** C'est la fonction de perte standard pour la classification binaire (minimiser l'erreur de probabilité).

**Q50 :** Si votre modèle XGBoost overfit, quels paramètres modifiez-vous en premier ?
**R50 :** Je réduis `max_depth`, j'augmente `min_child_weight`, ou j'augmente `gamma` (paramètre de régularisation).

---

## V. Deep Learning (PyTorch)

### 10. Architecture
**Q51 :** Pourquoi un MLP (Multi-Layer Perceptron) et pas un CNN ?
**R51 :** Les CNN sont faits pour les données spatiales (images) avec des corrélations locales. Nos données sont tabulaires, sans structure spatiale.

**Q52 :** Expliquez le rôle de `nn.BatchNorm1d`.
**R52 :** Il normalise les activations de la couche précédente (moyenne 0, variance 1). Cela stabilise le gradient, permet un learning rate plus élevé et accélère la convergence.

**Q53 :** Pourquoi utiliser l'activation `GELU` plutôt que `ReLU` ?
**R53 :** GELU (Gaussian Error Linear Unit) est une version plus lisse de ReLU. Elle permet souvent une meilleure convergence pour les réseaux profonds, bien que ReLU soit suffisant dans 90% des cas.

**Q54 :** À quoi sert le `Dropout(0.1)` ?
**R54 :** À éviter que les neurones ne co-adaptent trop. En éteignant aléatoirement 10% des neurones, on force le réseau à être plus robuste (redondance de l'information).

**Q55 :** Qu'est-ce qu'une "Skip Connection" (connexion résiduelle) et pourquoi l'avoir ajoutée ?
**R55 :** C'est ajouter l'entrée d'une couche directement à sa sortie ($x + f(x)$). Cela permet au gradient de circuler plus facilement lors de la rétropropagation, évitant le problème du "Vanishing Gradient" dans les réseaux profonds.

**Q56 :** Pourquoi avez-vous une couche `self.skip_proj` ?
**R56 :** Parce que la dimension d'entrée (128) était différente de la dimension de sortie (64). On ne peut pas additionner des vecteurs de tailles différentes, il faut donc projeter l'entrée pour qu'elle ait la même taille.

### 11. Entraînement
**Q57 :** Pourquoi utiliser l'optimiseur `Adam` ?
**R57 :** C'est un standard robuste. Il adapte le learning rate pour chaque paramètre individuellement en utilisant les moments du gradient.

**Q58 :** Qu'est-ce que le `OneCycleLR` ?
**R58 :** Une stratégie de Learning Rate qui commence bas, augmente jusqu'à un pic, puis redescend. Cela permet une convergence super rapide ("Super-convergence").

**Q59 :** Pourquoi convertir les données en `float32` pour PyTorch ?
**R59 :** Les GPU et les opérations matricielles sont optimisés pour le `float32`. Le `float64` (double) prend 2x plus de mémoire pour un gain de précision inutile en Deep Learning.

**Q60 :** Expliquez le concept d'Early Stopping.
**R60 :** On surveille la Loss sur le jeu de validation. Si elle ne baisse plus pendant `patience` époques, on arrête. Cela évite de continuer à entraîner un modèle qui commence à surapprendre.

**Q61 :** Pourquoi utiliser `nn.CrossEntropyLoss` pour une classification binaire ?
**R61 :** C'est standard. On aurait aussi pu utiliser `BCELoss` (Binary Cross Entropy) avec une seule sortie et une sigmoïde. `CrossEntropyLoss` attend 2 sorties (logits) et applique un Softmax interne.

**Q62 :** Que fait `loss.backward()` ?
**R62 :** C'est la rétropropagation (Backpropagation). Cela calcule le gradient de la perte par rapport à tous les poids du réseau.

**Q63 :** Que fait `optimizer.step()` ?
**R63 :** C'est la descente de gradient. Il met à jour les poids en soustrayant une fraction du gradient ($w = w - lr * grad$).

**Q64 :** Pourquoi `optimizer.zero_grad()` au début de la boucle ?
**R64 :** Parce que PyTorch accumule les gradients par défaut. Si on ne remet pas à zéro, on additionne les gradients de l'époque précédente, ce qui fausse tout.

**Q65 :** Qu'est-ce que le `Gradient Clipping` (`clip_grad_norm_`) ?
**R65 :** Si les gradients deviennent trop grands (Exploding Gradient), on les "coupe" à une valeur max (ici 1.0) pour éviter de déstabiliser l'entraînement.

**Q66 :** Pourquoi passer le modèle en `mlp.eval()` avant l'évaluation ?
**R66 :** Pour désactiver le Dropout et figer les statistiques de la Batch Normalization. En mode train, le Dropout est actif ; en mode eval, il est inactif.

**Q67 :** Votre réseau a-t-il fait mieux que XGBoost ?
**R67 :** (Réponse dépendante des résultats) Souvent, sur des données tabulaires de taille moyenne, XGBoost bat légèrement les réseaux de neurones ou fait jeu égal, mais avec beaucoup moins d'efforts de tuning.

**Q68 :** Comment auriez-vous pu améliorer le réseau ?
**R68 :** En utilisant une architecture dédiée au tabulaire comme **TabNet** ou en augmentant la taille du dataset (Data Augmentation).

---

## VI. Évaluation et Métriques

### 12. Métriques
**Q69 :** Définissez la Précision et le Rappel.
**R69 :**
*   Précision : Parmi ceux que j'ai prédits "Positifs", combien le sont vraiment ? (Qualité de la prédiction positive).
*   Rappel : Parmi tous les "Vrais Positifs" qui existent, combien en ai-je trouvé ? (Quantité détectée).

**Q70 :** Pourquoi le F1-score est-il une moyenne harmonique et pas arithmétique ?
**R70 :** La moyenne harmonique pénalise fortement les valeurs faibles. Si Rappel=100% mais Précision=0%, la moyenne arithmétique donne 50%, mais la moyenne harmonique donne 0%. Le F1 exige que les deux soient bons.

**Q71 :** Dans le cas d'un prêt bancaire, préférez-vous optimiser la Précision ou le Rappel ?
**R71 :** On veut éviter les défauts de paiement (Classe 1 = Défaut ? ou Classe 1 = Accord ? Attention au sens). Si Classe 1 = "Bon payeur", on veut une haute Précision (être sûr qu'il va payer). Si Classe 1 = "Défaut", on veut un haut Rappel (ne rater aucun mauvais payeur).

**Q72 :** Qu'est-ce que l'AUC-ROC ?
**R72 :** L'aire sous la courbe ROC. Elle mesure la capacité du modèle à classer un exemple positif aléatoire plus haut qu'un exemple négatif aléatoire. C'est indépendant du seuil de décision.

**Q73 :** Si AUC = 0.5, que vaut le modèle ?
**R73 :** Il est équivalent à un tirage à pile ou face (aléatoire).

**Q74 :** Qu'est-ce que la courbe Precision-Recall ? Quand l'utiliser ?
**R74 :** Elle trace la Précision en fonction du Rappel. Elle est préférable à la courbe ROC quand les classes sont très déséquilibrées, car elle ne prend pas en compte les Vrais Négatifs (qui sont majoritaires et gonflent artificiellement la ROC).

**Q75 :** Comment choisir le seuil de décision optimal ?
**R75 :** Par défaut c'est 0.5. On peut le déplacer pour favoriser le Rappel (ex: 0.3) ou la Précision (ex: 0.7) selon les coûts métier (Coût d'un Faux Positif vs Coût d'un Faux Négatif).

**Q76 :** Qu'est-ce qu'une matrice de confusion ?
**R76 :** Un tableau croisé qui montre les Vrais Positifs, Faux Positifs, Vrais Négatifs, Faux Négatifs. C'est la base de toutes les métriques.

**Q77 :** Pourquoi l'Accuracy est-elle trompeuse ici ?
**R77 :** Avec 80% de classe 0, un modèle "stupide" qui prédit toujours 0 aura 80% d'accuracy, mais sera inutile.

**Q78 :** Qu'est-ce que le "Macro Average" vs "Weighted Average" dans le classification report ?
**R78 :** Macro calcule la métrique pour chaque classe et fait la moyenne (donne autant d'importance à la classe minoritaire). Weighted pondère par le nombre d'exemples (donne plus d'importance à la classe majoritaire).

---

## VII. Mises en situation & "What If"

**Q79 :** Si le client vous dit "Je ne comprends pas pourquoi mon prêt est refusé", que faites-vous ?
**R79 :** J'utilise **SHAP** (SHapley Additive exPlanations) pour générer une explication locale : "Votre prêt est refusé car votre ratio dette/revenu contribue à -20% et votre âge à -5%".

**Q80 :** Si demain le dataset double de taille (100k lignes), quel modèle abandonnez-vous ?
**R80 :** Probablement le SVM (SVC) s'il était utilisé (complexité quadratique). Le Random Forest et XGBoost tiendront le coup.

**Q81 :** Si vous deviez déployer ce modèle sur un petit serveur avec peu de RAM, lequel choisiriez-vous ?
**R81 :** LightGBM ou HistGradientBoosting. Ils sont très optimisés pour la mémoire et la vitesse d'inférence.

**Q82 :** Si on vous demandait de prédire non pas "Oui/Non" mais le "Montant optimal à prêter", que changeriez-vous ?
**R82 :** Je passerais d'un problème de **Classification** à un problème de **Régression**. La target deviendrait `loan_amnt` (ou un montant ajusté) et la Loss function serait MSE (Mean Squared Error).

**Q83 :** Si vous découvrez que la variable `gender` est très discriminante, la gardez-vous ?
**R83 :** Question éthique et légale. Techniquement elle améliore le modèle, mais légalement (RGPD, lois anti-discrimination), il est souvent interdit de l'utiliser pour l'octroi de crédit. Je la supprimerais probablement pour éviter un biais discriminatoire ("Fairness").

**Q84 :** Votre modèle performe bien en dev mais mal en prod après 6 mois. Pourquoi ?
**R84 :** C'est du **Data Drift** (dérive des données). Le profil des clients a changé (ex: inflation, crise économique), et le modèle entraîné sur les données d'il y a 6 mois n'est plus adapté. Il faut réentraîner.

**Q85 :** Si vous aviez accès à des données textuelles (ex: motif du prêt écrit à la main), comment les intégreriez-vous ?
**R85 :** J'utiliserais du NLP (TF-IDF ou Embeddings type BERT) pour transformer le texte en vecteurs numériques, que j'ajouterais comme nouvelles features dans le modèle.

**Q86 :** Si vous deviez améliorer le score de 0.01 (1%), que tenteriez-vous en priorité ?
**R86 :** Du **Feature Engineering** métier (créer de nouvelles variables intelligentes) ou du **Stacking** (combiner les prédictions de XGBoost, RF et Neural Net).

**Q87 :** Comment valideriez-vous le modèle sans jeu de test ?
**R87 :** Impossible de valider rigoureusement sans données non vues. Je ferais une **Nested Cross-Validation** pour avoir une estimation robuste de l'erreur.

**Q88 :** Si le temps d'entraînement était limité à 1 minute, que feriez-vous ?
**R88 :** J'utiliserais LightGBM avec un `early_stopping` agressif et je réduirais le nombre d'arbres / la profondeur. Ou une simple Régression Logistique.

---

## VIII. Code & Technique

**Q89 :** Dans votre code PyTorch, pourquoi `X_train_t` est-il en `float32` et `y_train_t` en `long` ?
**R89 :** Les features (`X`) doivent être des flottants pour les multiplications matricielles. La target (`y`) pour `CrossEntropyLoss` doit être des entiers (`long` en PyTorch) représentant les indices des classes.

**Q90 :** À quoi sert `if __name__ == "__main__":` dans un script Python ?
**R90 :** À éviter que le code ne s'exécute si le fichier est importé comme module. C'est aussi indispensable pour le multiprocessing (utilisé par `n_jobs=-1` sous Windows/MacOS).

**Q91 :** Quelle est la différence entre `fit()` et `fit_transform()` ?
**R91 :** `fit()` calcule les paramètres (moyenne, écart-type). `transform()` applique la transformation. `fit_transform()` fait les deux d'un coup (plus efficace). On fait `fit_transform` sur le Train, et `transform` (seul) sur le Test.

**Q92 :** Pourquoi avoir utilisé `ColumnTransformer` ?
**R92 :** Pour appliquer des traitements différents aux colonnes numériques (Scaling) et catégorielles (OneHot) au sein d'un même pipeline scikit-learn.

**Q93 :** Dans le GridSearch, que signifie `cv=5` ?
**R93 :** Cross-Validation à 5 plis (folds). Le train set est divisé en 5. On entraîne sur 4, on valide sur 1. On répète 5 fois. Le score final est la moyenne des 5 scores.

**Q94 :** Pourquoi `verbose=-1` dans LightGBM ?
**R94 :** Pour supprimer les logs et warnings intempestifs lors de l'entraînement, gardant la sortie console propre.

**Q95 :** Comment sauvegarderiez-vous votre modèle pour le mettre en prod ?
**R95 :** Avec `joblib.dump(pipeline, 'model.pkl')` ou `pickle`. Pour PyTorch, `torch.save(model.state_dict(), 'model.pth')`.

**Q96 :** Quelle librairie avez-vous utilisée pour lire le CSV ?
**R96 :** Pandas (`pd.read_csv`).

**Q97 :** Comment avez-vous géré le téléchargement du fichier depuis Google Drive ?
**R97 :** Avec la librairie `gdown`, qui permet de télécharger des fichiers Drive via leur ID directement dans le notebook.

**Q98 :** Pourquoi `random_state=42` partout ?
**R98 :** Pour la reproductibilité. Cela fixe la graine aléatoire (seed) pour que les splits et les initialisations soient toujours les mêmes à chaque exécution.

**Q99 :** Dans le code PyTorch, pourquoi `input_dim` est-il dynamique ?
**R99 :** `input_dim = X_train.shape[1]`. Cela permet au réseau de s'adapter automatiquement si on ajoute ou enlève des features (ex: via OneHotEncoding) sans devoir réécrire le code.

**Q100 :** Avez-vous utilisé des environnements virtuels ?
**R100 :** Oui (ou "J'aurais dû"), pour isoler les dépendances du projet et éviter les conflits de versions entre bibliothèques (ex: numpy vs pandas).

**Q101 :** Qu'est-ce que le fichier `requirements.txt` ?
**R101 :** La liste de toutes les dépendances et leurs versions, permettant à quelqu'un d'autre de réinstaller exactement le même environnement avec `pip install -r requirements.txt`.
