# Full Analysis Code for User's Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("C:/Users/asus/Desktop/python toolbox/python project visualization/9ef84268-d588-465a-a308-a864a43d0070.csv")

# Clean column names
df.columns = df.columns.str.strip()
df.rename(columns={
    'Min_x0020_Price': 'MinPrice',
    'Max_x0020_Price': 'MaxPrice',
    'Modal_x0020_Price': 'ModalPrice',
    'Arrival_x0020_Date': 'Arrival_Date'
}, inplace=True)

# Convert Arrival_Date to datetime
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')

# Handle missing values
df['MinPrice'].fillna(df['MinPrice'].mean(), inplace=True)
df['MaxPrice'].fillna(df['MaxPrice'].mean(), inplace=True)
df['ModalPrice'].fillna(df['ModalPrice'].mean(), inplace=True)
df['Arrival_Date'].fillna(method='ffill', inplace=True)
df.drop_duplicates(inplace=True)

# Normalize MaxPrice and ModalPrice
scaler = MinMaxScaler()
df[['MaxPrice', 'ModalPrice']] = scaler.fit_transform(df[['MaxPrice', 'ModalPrice']])

# Objective 1: Top 10 commodities
commodity_counts = df['Commodity'].value_counts().head(10)
plt.figure(figsize=(8, 8))
plt.pie(commodity_counts.values, labels=commodity_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 10 Commodities Distribution")

# Objective 2: Top 10 markets by average modal price
top_markets = df.groupby('Market')['ModalPrice'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_markets.values, y=top_markets.index, palette="magma")
plt.title("Top 10 Markets by Avg Modal Price")

# Objective 3: Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df[['MinPrice','MaxPrice','ModalPrice']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")

# Objective 4: Histogram for top 5 commodities
top_commodities = df['Commodity'].value_counts().nlargest(5).index
df_top_commodities = df[df['Commodity'].isin(top_commodities)]
plt.figure(figsize=(10,6))
sns.histplot(data=df_top_commodities, x='ModalPrice', hue='Commodity', kde=True, bins=30)
plt.title("Distribution of Modal Prices - Top 5 Commodities")

# Objective 5: State-wise entry count
state_counts = df['State'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(x=state_counts.index, y=state_counts.values, palette='cubehelix')
plt.title("Record Count per State")
plt.xticks(rotation=90)


# Objective 6: Pairplot of price columns
sns.pairplot(df[['MinPrice','MaxPrice','ModalPrice']])
plt.suptitle("Pairwise Plot of Price Variables", y=1.02)
plt.tight_layout()

# Objective 7.  average Modal Price for each Commodity
plt.figure(figsize=(12, 6))
avg_modal = df.groupby('Commodity')['ModalPrice'].mean().sort_values(ascending=False).head(15)
sns.barplot(x=avg_modal.values, y=avg_modal.index, palette='crest')
plt.title("Average Modal Price by Commodity")

# Objective 8. Analyze how prices differ by Market
plt.figure(figsize=(14, 6))
top_markets = df['Market'].value_counts().head(10).index
sns.boxplot(data=df[df['Market'].isin(top_markets)], x='Market', y='ModalPrice')
plt.xticks(rotation=45)
plt.title("Modal Price Distribution by Market")

# Objective 9: Linear Regression
#linear Regression model (Max → Modal)
x = df[["MaxPrice"]];
y = df["ModalPrice"];
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
model = LinearRegression();
model.fit(x_train, y_train);

#Pridect
newdata = pd.DataFrame(({"MaxPrice": [0.75]}))  # any normalized value between 0-1
result = model.predict(newdata)
print("Pridiction: ", result)

#regression line plot
plt.figure()
plt.scatter(x, y, color="pink")
plt.plot(x, model.predict(x), color="blue", linewidth=3)
plt.xlabel("Max Price")
plt.ylabel("Modal Price")
plt.title("Linear Regression fit")
plt.grid(True)

#Mean Square Error
y_err = model.predict(x_test)
mse = mean_squared_error(y_test, y_err)
print("MSE: ", mse)
