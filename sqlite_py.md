YouTube SQL Analysis Using Python + SQLite

Step 1: Set Up SQLite and Load the Cleaned CSV


```python
import pandas as pd
import sqlite3

# Load your cleaned CSV
df = pd.read_csv("USvideos_cleaned.csv")

# Create SQLite database connection
conn = sqlite3.connect("youtube_trending.db")

# Save DataFrame as table in SQLite
df.to_sql("usvideos", conn, index=False, if_exists="replace")

```




    40875





Step 2: Create a Category Mapping Table from JSON


```python
import json

# Load category JSON file
with open("US_category_id.json") as f:
    category_data = json.load(f)

# Extract category_id and category_name from assignable items
categories = [
    (int(item["id"]), item["snippet"]["title"])
    for item in category_data["items"]
    if item["snippet"]["assignable"]
]

# Create DataFrame
category_df = pd.DataFrame(categories, columns=["category_id", "category_name"])

# Save to SQLite
category_df.to_sql("categories", conn, index=False, if_exists="replace")

```




    15





Step 3: Query – Rank Categories by Avg Views


```python
query1 = """
SELECT 
    c.category_name,
    ROUND(AVG(u.views), 2) AS avg_views
FROM 
    usvideos u
JOIN 
    categories c ON u.category_id = c.category_id
GROUP BY 
    c.category_name
ORDER BY 
    avg_views DESC;
"""

avg_views_df = pd.read_sql_query(query1, conn)
avg_views_df.to_csv("avg_views_by_category.csv", index=False)
avg_views_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category_name</th>
      <th>avg_views</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Music</td>
      <td>6204776.02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Film &amp; Animation</td>
      <td>3103241.46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nonprofits &amp; Activism</td>
      <td>2963884.07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gaming</td>
      <td>2607597.77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Entertainment</td>
      <td>2068105.02</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
