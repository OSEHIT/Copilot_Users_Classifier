# Copilot_Users_Classifier
# 🕵️‍♂️ Projet Copilote : L'art de vous reconnaître à vos clics

Imaginez-vous essayer de reconnaître un artiste non pas en voyant ses peintures, mais en analysant uniquement les traces de pinceau laissées sur sa palette. C'est le défi de ce projet !

L'objectif est de développer un modèle de classification capable d'identifier un utilisateur (`util`) en se basant uniquement sur la séquence brute de ses actions lors d'une session sur un logiciel.

## 📈 L'histoire de notre enquête : le Feature Engineering

Ce projet est une histoire en deux temps : une première tentative basée sur des généralités, et une seconde qui a compris que... le diable est dans les détails.

### Acte 1 : L'approche "Comptons les actions"

Mon premier instinct face à un fichier `train.csv` contenant plus de 7000 actions uniques ? Simplifier !

1.  **Le grand nettoyage :** J'ai regroupé ces 7000+ actions en 35 catégories de base (ex: "Création d'écran", "Double-clic", "Lancement d'une stat").
2.  **L'analyse des habitudes :** J'ai compté les occurrences de chaque catégorie. Pour aller plus loin, j'ai aussi utilisé des **bigrammes** (ex: "Affichage dialogue" → "Exécution bouton") avec `TfidfVectorizer` pour capturer les *séquences* d'actions.
3.  **Premiers résultats :** J'ai testé plusieurs modèles (`SVC`, `MLP`, `LogisticRegression`). Le `RandomForestClassifier` a largement dominé, atteignant un F1-score (macro) d'environ **0.76**.

C'était un bon début, mais je sentais qu'on passait à côté de l'essentiel.

### Acte 2 : L'illumination "Le 'où' compte plus que le 'quoi'"

**L'hypothèse :** Un "Double-clic" dans le module de *facturation* n'est pas le même qu'un "Double-clic" dans le *CRM*. L'identité de l'utilisateur ne réside pas dans l'action "cliquer", mais dans *sur quoi* il clique.

1.  **La nouvelle stratégie :** J'ai abandonné le regroupement simpliste. À la place, j'ai utilisé des expressions régulières (Regex) pour extraire les *patterns* spécifiques des logs bruts :
    * Les écrans (ex: `infologic.core.accueil...`)
    * Les configurations (ex: `MAINT`, `DEF_03/24`)
    * Les chaînes (ex: `GP`, `ST`)
2.  **La "signature" TF-IDF :** J'ai traité tous les patterns d'une session comme un seul "document" et j'ai appliqué `TfidfVectorizer` (avec 1000 features max). Cela a créé une véritable "signature logicielle" pour chaque session, basée sur les modules spécifiques que l'utilisateur fréquente.
3.  **Le résultat :** En ré-entraînant le `RandomForestClassifier` avec ces nouveaux features (en plus des comptages et bigrammes de l'Acte 1), le F1-score (macro) est monté en flèche pour atteindre **0.88** !

**Conclusion :** Ce n'est pas seulement *comment* vous cliquez qui vous définit, c'est *où* vous cliquez dans le logiciel.

## 🏆 Le Modèle Champion

Le **RandomForestClassifier** (n_estimators=300) a été le choix final. Il excelle à gérer des données "sparse" (comme le TF-IDF) et à trouver des relations complexes que d'autres modèles (comme la Régression Logistique ou les SVM) ont manquées.

Le pipeline de feature engineering complet de l'Acte 2 a été appliqué au fichier `test.csv` pour générer le fichier de soumission `submission.csv`.

## 🛠️ Comment l'exécuter ?

Envie de voir la magie opérer ?

1.  Assurez-vous d'avoir Python et Git installés.
2.  (Recommandé) Créez un environnement virtuel pour garder les choses propres :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```
3.  Installez les outils nécessaires listés dans `requirements.txt` :
    ```bash
    pip install -r requirements.txt
    ```
4.  Lancez Jupyter et ouvrez le notebook :
    ```bash
    jupyter notebook "Notebook DS.ipynb"
    ```
