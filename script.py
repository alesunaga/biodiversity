import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
observations = pd.read_csv('observations.csv')
species = pd.read_csv('species_info.csv')

# Replace the missing values in the `conservation_status` column with `No Intervention`
species.conservation_status = species.conservation_status.fillna('No Intervention')

# Merge the `observations` and `species` dataframes
merged_df = pd.merge(observations, species, on='scientific_name')

# Create a pivot table to aggregate the `observations` for each `park_name` and `is_protected` combination
pivot_table = merged_df.pivot_table(
    index='park_name',
    columns='conservation_status',
    values='observations'
)

# Create a stacked bar chart to visualize the pivot table
pivot_table.plot(kind='bar', stacked=True)

# Use `park_name` for the x-axis, `observations` for the y-axis, and `is_protected` for the color
plt.xlabel('Park Name')
plt.ylabel('Average Observations')

# Set the title to 'Average Observations by Park and Protection Status' and display the plot
plt.title('Average Observations by Park and Protection Status')
plt.show()

# Create a new dataframe `species_without_no_intervention` by filtering out the rows where `conservation_status` is 'No Intervention'
species_without_no_intervention = species[species.conservation_status != 'No Intervention']

# Create a bar chart to visualize the distribution of conservation status for animals using seaborn
sns.countplot(x='conservation_status', data=species_without_no_intervention)

# Set the title to 'Conservation Status by Species' and display the plot
plt.title('Conservation Status by Species')
plt.show()

# Create a subdataframe for the conservation status 'Species of Concern'
species_of_concern = species[species.conservation_status == 'Species of Concern']

# Create a barplot using seaborn with `category` on the x-axis and the count on the y-axis
sns.countplot(x='category', data=species_of_concern)

# Rotate the x-axis labels by 45 degrees to avoid overlap
plt.xticks(rotation=45)

# Set the title to 'Distribution of Species of Concern' and display the plot
plt.title('Distribution of Species of Concern')
plt.show()

# Merge the `observations` and `species` dataframes on the `scientific_name` column
merged_df = pd.merge(observations, species, on='scientific_name')

# Filter the merged dataframe to only include rows where `conservation_status` is 'Species of Concern'
species_of_concern_df = merged_df[merged_df.conservation_status == 'Species of Concern']

# Create a pivot table to aggregate the `observations` for each `park_name` and `category` combination
species_of_concern_by_park = species_of_concern_df.pivot_table(
    index='park_name',
    columns='category',
    values='observations'
)

# Increase the size of the figure
plt.figure(figsize=(12, 10))

# Create a heatmap using seaborn with `species_of_concern_by_park` as the data
sns.heatmap(species_of_concern_by_park, annot=True, cmap='YlGnBu', annot_kws={'size': 12}, fmt='.0f')

# Set the title and display the plot
plt.title('Average Observations of Species of Concern by Park and Category')
plt.show()

# Perform statistical analysis
contingency_table = pd.crosstab(species['conservation_status'], species['category'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
std_residuals = (contingency_table - expected) / np.sqrt(expected)
z_scores = norm.ppf(1 - 0.05 / 2)
significant_cells = std_residuals[abs(std_residuals) > z_scores]

# Print the results
print(f'Chi-squared statistic: {chi2}')
print(f'P-value: {p_value}')
print(f'Degrees of freedom: {dof}')
print('Standardized residuals:')
print(std_residuals)
print('Significant deviations:')
print(significant_cells)

# Save the figures
plt.savefig('Conservation Status by Species.png')
plt.savefig('Average Observations of Species of Concern by Park and Category.png', dpi=300)
