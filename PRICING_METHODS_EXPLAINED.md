# 📊 Différence entre les 3 Méthodes de Pricing BTP

## 🎯 Vue d'Ensemble

| Méthode | Source de Données | Avantages | Inconvénients |
|---------|------------------|-----------|---------------|
| **method_db_standard** | Catalogue national (Batiprix) | Rapide, toujours disponible | Générique, ignore contexte |
| **method_knn_historical** | Projets passés de l'entreprise | Précis, contextuel | Besoin d'historique |
| **method_regional_adjust** | Catalogue + facteurs régionaux | Bon compromis | Facteurs fixes (pas adaptatifs) |

---

## 📘 Méthode 1 : DB Standard (Catalogue National)

### Source
```
Batiprix 2024 (catalogue officiel français)
ISBN, organismes de construction
Mis à jour 1-2 fois par an
```

### Exemple de Catalogue
```python
BATIPRIX_2024 = {
    "concrete_wall": 250€/m²,  # Prix moyen NATIONAL
    "brick_facade": 180€/m²,
    "wood_floor": 200€/m²
}
```

### Calcul
```python
# Estimer : Béton 20m² à Paris
prix_catalogue = 250€/m²
estimation = 250€ × 20m² = 5000€
```

### Problème
❌ Ignore que Paris est 10-15% plus cher
❌ Ne sait pas que ton fournisseur fait -5%
❌ Catalogue publié en janvier, on est en octobre (prix ont évolué)

**Résultat :** 5000€ (sous-estimation pour Paris)

---

## 🗄️ Méthode 2 : KNN sur Projets Passés (Historique Entreprise)

### Source
```
Base de données interne de TON entreprise
Projets RÉELLEMENT terminés avec factures finales
Mise à jour après chaque projet
```

### Exemple d'Historique
```python
COMPLETED_PROJECTS = [
    {
        "project": "École Lille 2023",
        "material": "concrete",
        "surface": 20,
        "region": "north",
        "final_price": 5000€  # Prix RÉEL payé
    },
    {
        "project": "Lycée Paris 2024",
        "material": "concrete",
        "surface": 20,
        "region": "paris",
        "final_price": 5500€  # Prix RÉEL payé (premium Paris)
    },
    {
        "project": "Collège Paris 2023",
        "material": "concrete",
        "surface": 25,
        "region": "paris",
        "final_price": 6800€  # Prix RÉEL payé
    }
]
```

### Algorithme KNN
```python
# Estimer : Béton 20m² à Paris

# 1. Chercher projets similaires (même matériau + région)
candidates = filter(COMPLETED_PROJECTS, material="concrete", region="paris")
# → Trouve : 20m² = 5500€, 25m² = 6800€

# 2. Calculer distances
distance_1 = |20 - 20| = 0  (exact match!)
distance_2 = |20 - 25| = 5

# 3. Moyenne pondérée (inverse distance)
weight_1 = 1/(0+1) = 1.0
weight_2 = 1/(5+1) = 0.167

estimation = (1.0×5500 + 0.167×6800) / (1.0+0.167)
estimation = 5600€
```

### Avantages
✅ Sait que Paris = +10% (historique le montre)
✅ Inclut tes négociations fournisseurs
✅ Prix réels, pas moyennes théoriques

**Résultat :** 5600€ (précis pour Paris)

---

## 🌍 Méthode 3 : Regional Adjust (Catalogue + Facteurs)

### Source
```
Catalogue national (250€/m²)
+ Facteurs régionaux fixes
```

### Facteurs Régionaux
```python
REGIONAL_MULTIPLIERS = {
    "north": 1.0,    # Prix normal
    "south": 0.95,   # 5% moins cher
    "paris": 1.15    # 15% plus cher
}
```

### Calcul
```python
# Estimer : Béton 20m² à Paris
prix_base = 250€ × 20m² = 5000€
facteur_paris = 1.15
estimation = 5000€ × 1.15 = 5750€
```

### Caractéristiques
✅ Prend en compte les régions
❌ Facteurs fixes (ne s'adaptent pas)
❌ Ne sait pas si ton fournisseur parisien fait -5%

**Résultat :** 5750€ (surestimation légère)

---

## 📊 Comparaison sur un Exemple Concret

### Projet : Mur Béton 20m² à Paris

| Méthode | Estimation | Erreur vs Réel (5500€) | Explication |
|---------|-----------|------------------------|-------------|
| **DB Standard** | 5000€ | -500€ (-9%) | Ignore premium Paris |
| **KNN Historical** | 5600€ | +100€ (+2%) | Très précis (historique Paris) |
| **Regional Adjust** | 5750€ | +250€ (+5%) | Facteur Paris trop élevé (1.15 au lieu de 1.10) |

**Gagnant :** KNN Historical ✅

---

## 🎓 Quand Utiliser Quelle Méthode ?

### DB Standard (Catalogue)
**Bon pour :**
- ✅ Éléments peu communs (pas d'historique)
- ✅ Estimations rapides/rough
- ✅ Régions stables (pas Paris)

**Mauvais pour :**
- ❌ Régions volatiles (Paris, grandes villes)
- ❌ Estimations précises
- ❌ Projets critiques (offres concurrentielles)

---

### KNN Historical (Projets Passés)
**Bon pour :**
- ✅ Éléments standards (beaucoup d'historique)
- ✅ Régions où tu as déjà travaillé
- ✅ Estimations précises pour appels d'offres

**Mauvais pour :**
- ❌ Nouveaux matériaux (pas d'historique)
- ❌ Nouvelles régions (première fois)
- ❌ Surfaces extrêmes (très petit/grand)

---

### Regional Adjust (Facteurs)
**Bon pour :**
- ✅ Compromis vitesse/précision
- ✅ Régions connues mais peu d'historique
- ✅ Estimations moyennement précises

**Mauvais pour :**
- ❌ Marchés très volatils
- ❌ Projets critiques
- ❌ Conditions exceptionnelles

---

## 🤖 Ce que l'Agent RL Apprend

Après 20 itérations, l'agent découvre automatiquement :

```
Context: Béton 20m² Nord
→ method_db_standard (Q=0.75)  ⭐ Meilleur choix
   (Prix stable, pas de premium régional)

Context: Béton 20m² Paris
→ method_knn_historical (Q=0.85)  ⭐ Meilleur choix
   (Historique parisien précis)

Context: Béton 50m² Paris
→ method_knn_historical (Q=0.90)  ⭐ Meilleur choix
   (KNN excellent pour grandes surfaces avec historique)

Context: Bois 15m² Sud
→ method_db_standard (Q=0.72)  ⭐ Meilleur choix
   (Région stable, peu d'historique bois)
```

**Conclusion :** L'agent apprend **automatiquement** quelle méthode utiliser selon le contexte, sans règles hardcodées ! 🚀

---

## 💡 En Production Réelle

### DB Standard
```python
# Connexion API Batiprix
import batiprix_api

def method_db_standard(material, surface):
    catalog_price = batiprix_api.get_price(material, year=2024)
    return catalog_price * surface
```

### KNN Historical
```sql
-- Requête SQL sur DB interne
SELECT material, surface, region, final_price, project_date
FROM completed_projects
WHERE material = 'concrete'
  AND region = 'paris'
  AND project_date > '2023-01-01'
ORDER BY ABS(surface - 20)  -- Distance
LIMIT 3  -- K=3 neighbors
```

### Regional Adjust
```python
# API facteurs régionaux (INSEE, etc.)
import insee_api

def method_regional_adjust(material, surface, region):
    base_price = CATALOG[material] * surface
    regional_factor = insee_api.get_price_index(region, year=2024)
    return base_price * regional_factor
```

---

**🎯 Maintenant tu comprends la différence !**
