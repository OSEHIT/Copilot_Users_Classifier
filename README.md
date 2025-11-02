# Copilot_User_Classifier
# Projet Copilote : Classification Comportementale des Utilisateurs de Copilote

Ce projet de Data Science présente une méthodologie pour l'identification d'utilisateurs (variable `util`) basée sur l'analyse de leurs séquences d'actions et de leur comportement en session.

L'objectif est de développer un modèle de classification multi-classe robuste, capable de prédire l'identité d'un utilisateur à partir de ses journaux (logs) d'interaction bruts (`train.csv`, `test.csv`).

## 📈 Approche Méthodologique et Feature Engineering

La performance de ce type de classification repose presque entièrement sur la qualité de l'ingénierie des caractéristiques (feature engineering). L'analyse a donc suivi une démarche itérative pour raffiner progressivement la représentation des données de session.

### Itération 1 : Modélisation par Fréquence d'Actions

La première approche a consisté à agréger les actions pour réduire la dimensionnalité.

1.  **Réduction des Caractéristiques :** Les >7000 actions uniques ont été normalisées et regroupées en 35 actions génériques (ex: "Création d'écran", "Double-clic", "Lancement d'une stat").
2.  **Extraction de Features :**
    * **Fréquence :** Comptage des occurrences de chaque action générique par session.
    * **Séquentiel (N-grammes) :** Création de bigrammes (ex: "Action A → Action B") et application de `TfidfVectorizer` pour capturer les transitions d'actions communes.
    * **Métadonnées :** Ajout du `temps total` de session (calculé via les marqueurs `tXX`) et de la `first_op` / `last_op`.
3.  **Évaluation Initiale :** Plusieurs modèles ont été évalués (via F1-score macro) :
    * **`RandomForestClassifier` : ~0.76**
    * `LogisticRegression` : ~0.58
    * `MLPClassifier` : ~0.51
    * `SVC` (divers noyaux) : ~0.12-0.36

Le `RandomForestClassifier` s'est montré le plus performant, mais le score de 0.76 indiquait une perte d'information significative due à l'agrégation des actions.

### Itération 2 : Modélisation par Analyse Sémantique (TF-IDF)

**Hypothèse de raffinement :** L'identité de l'utilisateur ne réside pas dans l'action générique (ex: "Clic"), mais dans son contexte spécifique (ex: "Clic sur *infologic.crm.modules.CRM_COMPTE*").

1.  **Conservation du Contexte :** Au lieu de regrouper les actions, des expressions régulières (Regex) ont été utilisées pour extraire les *patterns* sémantiques bruts des logs :
    * Écrans (ex: `infologic.core.accueil...`)
    * Configurations (ex: `MAINT`, `DEF_03/24`)
    * Chaînes (ex: `GP`, `ST`)
2.  **Création d'une "Signature" TF-IDF :** L'ensemble des patterns d'une session a été traité comme un "document". Un `TfidfVectorizer` (avec `max_features=1000`) a été appliqué pour transformer ces patterns en une "signature comportementale" numérique, capturant les modules et contextes les plus utilisés par chaque utilisateur.
3.  **Amélioration de la Performance :** En combinant les features de l'Itération 1 (comptages, bigrammes) avec cette nouvelle matrice TF-IDF, le `RandomForestClassifier` a atteint un **F1-score (macro) de 0.88** en validation (et 0.87 en validation croisée).

## 🏆 Modèle Final et Conclusion

Le modèle retenu est un **`RandomForestClassifier`** (n_estimators=300) entraîné sur l'ensemble de caractéristiques enrichi de l'Itération 2. Ce choix est justifié par sa capacité à gérer efficacement la haute dimensionnalité (plus de 1900 features) et la nature "sparse" (pleine de zéros) des données TF-IDF.

**Conclusion :** L'analyse démontre que l'identité d'un utilisateur est plus fortement corrélée aux *contextes* spécifiques de son interaction (les écrans et modules fréquentés) qu'à la simple *fréquence* de ses actions génériques.

Le pipeline de feature engineering complet a été appliqué au fichier `test.csv` pour générer le fichier de soumission final `submission.csv`.

## 🛠️ Installation et Utilisation

Pour répliquer cette analyse :

1.  **Cloner le dépôt** (si applicable) :
    ```bash
    git clone [URL_DU_REPO]
    cd [NOM_DU_PROJET]
    ```

2.  **(Recommandé) Créer un environnement virtuel** :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

3.  **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

4.  **Lancer Jupyter** :
    ```bash
    jupyter notebook "Notebook DS.ipynb"
    ```
    Ouvrez le notebook et exécutez les cellules.
