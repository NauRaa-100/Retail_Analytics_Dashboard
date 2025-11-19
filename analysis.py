
# ============================
#  1) Import Libraries
# ============================

import numpy as np
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ============================
#  2) Load Dataset
# ============================

df = pd.read_csv(r"Online_Retail.csv", encoding="latin")


# ============================
#  3) Basic Info
# ============================

print(df.head())
print(df.shape)
print(df.info())
print(df.describe().T)


# ============================
#  4) Remove Duplicates
# ============================

print("Duplicated rows:", df.duplicated().sum())
df = df.drop_duplicates()


# ============================
#  5) Handle Missing Values
# ============================

# Description (1454 missing) → fill with "Unknown"
df["Description"] = df["Description"].fillna("Unknown")

# CustomerID (135k missing) → VERY IMPORTANT
# "Guest Customers"
df["CustomerID"] = df["CustomerID"].fillna(0).astype(int)


# ============================
#  6) Convert InvoiceDate
# ============================

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)

df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Day"] = df["InvoiceDate"].dt.day
df["Hour"] = df["InvoiceDate"].dt.hour
df["Weekday"] = df["InvoiceDate"].dt.weekday


# ============================
#  7) Calculate Total Price
# ============================

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]


# ============================
#  8) Remove Negative Quantities (Returns)
# ============================

df_pos = df[df["Quantity"] > 0]


# ============================
#  9) Optimize Data Types
# ============================

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category")
    if str(df[col].dtype) == "float64":
        df[col] = df[col].astype("float32")
    if str(df[col].dtype) == "int64":
        df[col] = df[col].astype("int32")

print(df.info())

#---------
monthly_sales = df_pos.groupby("Month")["TotalPrice"].sum()
print(monthly_sales)

plt.figure(figsize=(10,5))
sns.barplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title("Total Sales Per Month")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.show()


#-----
best_month = monthly_sales.idxmax()
worst_month = monthly_sales.idxmin()

print("Best Month:", best_month)
print("Worst Month:", worst_month)

#-----
top_products = df_pos.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10)
print(top_products)

plt.figure(figsize=(12,5))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top 10 Best-Selling Products")
plt.xlabel("Total Revenue")
plt.ylabel("Product")
plt.show()

#-----

worst_products = df_pos.groupby("Description")["TotalPrice"].sum().sort_values().head(10)
print(worst_products)

#-------
country_sales = df_pos.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
print(country_sales)

plt.figure(figsize=(12,6))
country_sales.head(15).plot(kind='bar')
plt.title("Top Countries by Revenue")
plt.ylabel("Revenue")
plt.show()

#--------
margin_proxy = df_pos.groupby("Description")["UnitPrice"].median().sort_values(ascending=False).head(10)
print(margin_proxy)

#  Customer Behavior Analysis

#Recency = max(invoice_date) - last_purchase_date
#Frequency = number of unique invoices per customer


# --- Convert date column ---
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# --- Create TotalPrice ---
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# --- Latest date in dataset ---
max_date = df['InvoiceDate'].max()

# --- RFM Base Table ---
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                              # Frequency
    'TotalPrice': 'sum'                                  # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# OPTIONAL: filter negative or zero monetary (refunds)
rfm = rfm[rfm['Monetary'] > 0]

rfm.head()

customers_country = df.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False)
customers_country


top_customers = rfm.sort_values('Monetary', ascending=False).head(10)
top_customers

rfm['AvgSpend'] = rfm['Monetary'] / rfm['Frequency']
rfm[['CustomerID','AvgSpend']].head()

first_purchase = df.groupby('CustomerID')['InvoiceDate'].min()
returning = df.groupby('CustomerID')['InvoiceNo'].nunique()

customer_type = pd.DataFrame({
    'CustomerID': first_purchase.index,
    'FirstPurchase': first_purchase.values,
    'Orders': returning.values
})

customer_type['CustomerType'] = customer_type['Orders'].apply(lambda x: 'Returning' if x > 1 else 'New')
customer_type.head()


plt.hist(rfm['Recency'])
plt.title("Recency Distribution")
plt.xlabel("Days Since Last Purchase")
plt.ylabel("Customers Count")
plt.show()


plt.hist(rfm['Frequency'])
plt.title("Frequency Distribution")
plt.xlabel("Number of Purchases")
plt.ylabel("Customers Count")
plt.show()


customers_country.plot(kind='bar', figsize=(12,4))
plt.title("Customers by Country")
plt.show()


df.to_csv("new_Retail.csv", index=False)
print("Saved new_Retail.csv successfully!")


'''
Correlations -->
             Quantity  UnitPrice  Total Price
Quantity     1.000000  -0.012666     0.187511    
UnitPrice   -0.012666   1.000000    -0.528692    
Total Price  0.187511  -0.528692     1.000000    

'''

