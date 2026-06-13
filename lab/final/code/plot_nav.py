import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
df_weighted = pd.read_csv('back_test_hs300.csv')
df_equal = pd.read_csv('back_test_equal_hs300.csv')

# Convert date column to datetime
df_weighted['date'] = pd.to_datetime(df_weighted['date'])
df_equal['date'] = pd.to_datetime(df_equal['date'])

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_weighted['date'], df_weighted['nav'], label='Model Weights NAV', linewidth=1.5)
plt.plot(df_equal['date'], df_equal['nav'], label='Equal Weights NAV', linewidth=1.5)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Net Asset Value (NAV)', fontsize=12)
plt.title('NAV Comparison: Model Weights vs Equal Weights', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

# Save and show
plt.savefig('nav_comparison.png', dpi=300)
print("Plot successfully saved to nav_comparison.png")
