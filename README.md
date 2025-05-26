# 🎯 CrowdFunding Predictor: Kickstarter Success Model

A machine learning pipeline in R that predicts whether a Kickstarter campaign will successfully reach its fundraising goal based on campaign text, metadata, and reward structure.

---

### 📌 Project Overview

This project analyzes over **100,000 Kickstarter campaigns** from 2009–2014 to predict if a project will be **successful** (i.e., raises more than its funding goal). Each campaign includes 60+ features like description, tags, reward amount, category, sentiment scores, and launch timing.

---

### 🎯 Objective

The binary target variable `success` is:
- **YES** → Project raised more than its goal
- **NO** → Project did not meet its goal

We predict `success` using engineered features, textual cues, and classification models.

---

### 📊 Data Description

The dataset is based on real Kickstarter campaigns and includes information for over **100,000 projects** from **2009 to 2014**. It is structured into training and test splits, with each row representing a single crowdfunding campaign.

---

🔄 Dataset Breakdown

| File Name              | Description                                                           |
|------------------------|------------------------------------------------------------------------|
| `small_training_X.csv` | 10,000 Kickstarter projects (feature data) for training and exploration |
| `small_training_y.csv` | Labels (YES/NO) for whether each project was successful                |
| `ks_test_X.csv`        | Test data (without true labels) used for final prediction              |

---

🏷️ Target Variable

| Column     | Type   | Description                                       |
|------------|--------|---------------------------------------------------|
| `success`  | Binary | `YES` = Funded above goal, `NO` = Failed to meet goal |

---

### 🔧 Methodology

1️⃣ Data Preparation
- Loaded and joined training/testing datasets
- Converted timestamps and cleaned missing values

2️⃣ Feature Engineering
- Text metrics (avg sentence length, part of speech)
- Reward-based indicators (`reward_count`, `avg_reward_amount`)
- Sentiment analysis using `afinn`, `syuzhet`, `sentimentr`
- Visual features: smiling flag, image presence
- Project metadata: category, goal, region, launch timing
- Flags for project creators with multiple submissions

3️⃣ Modeling Techniques
- **Decision Trees** with rpart
- **Random Forest** with `ranger`
- **XGBoost** for boosting performance
- Feature selection using `FSelector`
- Applied **SMOTE** for class imbalance handling

4️⃣ Evaluation
- 10-fold cross-validation
- Accuracy and AUC used for comparison
- Final model tuned using hyperparameter grid search

---

### 📈 Results

- **Best model:** XGBoost
- **Best AUC:** 0.74 on validation
- **Best Accuracy:** 72.5%
- Significant features included sentiment polarity, reward structure, launch weekday, and project length

