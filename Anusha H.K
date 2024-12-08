import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.weightstats import ztest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data (replace "dummy_data.csv" with your actual CSV file path)
df = pd.read_csv("dummy_data.csv")

# Clean column names by stripping extra spaces and converting to lowercase
df.columns = df.columns.str.strip().str.lower()

# Check the first few rows and columns to inspect the dataset
print(df.head())
print(df.columns)

# Ensure 'platform' and 'time_spent' columns exist before proceeding
if 'platform' not in df.columns or 'time_spent' not in df.columns:
    raise KeyError("Required columns 'platform' or 'time_spent' are missing in the dataset.")

# 1. Bar Chart: Average time spent on each platform
avg_time_per_platform = df.groupby('platform')['time_spent'].mean()
avg_time_per_platform.sort_values().plot(kind='bar', figsize=(8, 5), color='lightblue', edgecolor='black')
plt.title('Average Time Spent on Each Platform')
plt.ylabel('Average Time Spent')
plt.xlabel('Platform')
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Pie Chart: Distribution of Gender
if 'gender' in df.columns:
    gender_distribution = df['gender'].value_counts()
    gender_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90, figsize=(6, 6), colors=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Gender Distribution')
    plt.ylabel('')  # Hide the y-label for aesthetics
    plt.show()
else:
    print("Column 'gender' is missing in the dataset.")

# 3. Histogram: Age Distribution
if 'age' in df.columns:
    df['age'].plot(kind='hist', bins=20, figsize=(8, 5), color='mediumpurple', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Column 'age' is missing in the dataset.")

# 4. Donut Chart: Average Time Distribution by Profession
if 'profession' in df.columns and 'time_spent' in df.columns:
    avg_time_by_profession = df.groupby('profession')['time_spent'].mean()
    avg_time_by_profession.sort_values(inplace=True)

    # Create a pie chart with a hole in the center
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        avg_time_by_profession,
        labels=avg_time_by_profession.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("pastel"),
        pctdistance=0.85
    )
    
    # Draw the donut hole
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gca().add_artist(center_circle)

    # Add title and adjust layout
    plt.title('Average Time Distribution by Profession', fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("Columns 'profession' or 'time_spent' are missing in the dataset.")

# 5. Line Plot: Relationship between Time Spent and Income
if 'time_spent' in df.columns and 'income' in df.columns:
    df.groupby('time_spent')['income'].mean().plot(kind='line', figsize=(8, 5), color='green', marker='o')
    plt.title('Income Trend with Time Spent')
    plt.xlabel('Time Spent')
    plt.ylabel('Average Income')
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Columns 'time_spent' or 'income' are missing in the dataset.")

# Hypothesis Testing

# T-test: To compare average income between different professions
profession_groups = df['profession'].unique()

# Ensure there are at least two professions to compare
if len(profession_groups) >= 2:
    group1 = df[df['profession'] == profession_groups[0]]['income']
    group2 = df[df['profession'] == profession_groups[1]]['income']
    t_stat, p_val = stats.ttest_ind(group1, group2)
    print(f'T-test results between {profession_groups[0]} and {profession_groups[1]}:')
    print(f't-statistic: {t_stat}, p-value: {p_val}')
else:
    print("Not enough distinct professions for T-test.")

# Future Prediction: Average Time on Social Media
# Column Graph: Predicted average time spent in future years
current_avg_time = 143  # minutes per day
growth_rate = 3  # minutes per year
years = [2023, 2024, 2025, 2026]

future_avg_times = [current_avg_time + (growth_rate * year) for year in years]

# Plot Column Graph for Predicted Time
plt.figure(figsize=(8, 5))
plt.bar(years, future_avg_times, color='skyblue', edgecolor='black', width=3)
plt.title('Predicted Average Time Spent on Social Media')
plt.xlabel('Years in Future')
plt.ylabel('Average Time Spent (minutes/day)')
plt.xticks(years)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print predicted values for reference
for year, avg_time in zip(years, future_avg_times):
    print(f"Predicted average time spent on social media in {year} years: {avg_time} minutes per day")

