from datasets import load_dataset

print("Loading dataset...")

dataset = load_dataset("mdonigian/iab-news-classification")

print(dataset)

df = dataset["train"].to_pandas()

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nCategories:")
print(df["iab_category"].value_counts())
