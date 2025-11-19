import pandas as pd
from analysis import df
from mlxtend.frequent_patterns import apriori, association_rules

basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)

# Convert quantities to 1/0 (presence/absence)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

basket.head()


# Frequent itemsets
frequent_items = apriori(basket, min_support=0.02, use_colnames=True)

frequent_items.sort_values('support', ascending=False).head()

rules = association_rules(frequent_items, metric="lift", min_threshold=1)

# Sort strongest rules
rules = rules.sort_values('lift', ascending=False)

rules.head(10)

rules.head(10)[['antecedents','consequents','support','confidence','lift']]

import matplotlib.pyplot as plt

plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.title('Association Rules')
plt.show()

rules.to_csv("basket_rules.csv", index=False)
print("Saved basket_rules.csv successfully!")
