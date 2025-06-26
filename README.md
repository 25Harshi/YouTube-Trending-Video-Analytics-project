# 📊 YouTube-Trending-Video-Analytics-project

This project explores real-world YouTube trending video data to identify patterns in viewer engagement, publishing behavior, category performance, and sentiment trends. Using Python, SQLite, and Tableau, I’ve built a full-stack analytics workflow with interactive dashboards and data storytelling.

---

## 🎯 Objective
To uncover patterns in trending videos by analyzing YouTube datasets—focusing on visibility factors such as category, publish time, engagement, and tags.

---

## 🛠 Tools & Technologies

- **Python** (Pandas, Matplotlib, Seaborn, TextBlob) – Data cleaning, feature engineering, sentiment analysis, visualizations  
- **SQLite** – SQL queries for category-wise insights and average view ranking  
- **Tableau** – Interactive dashboards and visual storytelling  
- **Excel** – Initial data formatting and inspection  

---

## 📁 Dataset Overview

- **CSV:** `USvideos_cleaned_final.csv` (cleaned YouTube trending data for the US)  
- **JSON:** `US_category_id.json` (category mapping file)  
- ~40,000 rows with metadata including views, likes, publish time, category ID, etc.

---

## 🔄 Project Workflow

### 1. Data Cleaning & Preprocessing
- Removed duplicates and missing descriptions  
- Converted `publish_time` to datetime  
- Added `publish_hour`, `publish_day`, `likes/views`, `trending_days`  
- Mapped `category_id` → `category_name` using JSON

### 2. Exploratory Data Analysis (Python)
- Views vs Trending Duration  
- Category-wise engagement  
- Word cloud for most common tags  
- Sentiment distribution of video titles

### 3. Sentiment Analysis
- Used `TextBlob` on video titles  
- Classified as Positive, Neutral, or Negative  
- Visualized sentiment impact on views

### 4. SQL (SQLite via Python)
- Queried average views by category  
- Analyzed publish hour vs viewership  
- Exported query results to CSV for dashboard use

### 5. Dashboard (Tableau)
- Bar charts, donut charts, scatter plots, KPI cards  
- Filters: Category, Sentiment, Publish Hour  
- Interactive panels and storytelling layout

---



## 📌 Key Insights

•	Entertainment & Music dominate in visibility and average views
•	Publishing time impacts success—midweek + mid-morning = higher views
•	Positive & neutral sentiment boosts user interaction
•	Viewer trust is reflected through high likes/view ratios
•	Trending duration helps—but content quality plays a greater role


---

## ✅ Project Highlights

- Cleaned and transformed real-world YouTube data  
- Used sentiment analysis to enrich feature space  
- Performed SQL-based ranking inside Python via SQLite  
- Built an interactive Tableau dashboard with KPI tracking  
- Designed for data storytelling and decision-making insight

---


