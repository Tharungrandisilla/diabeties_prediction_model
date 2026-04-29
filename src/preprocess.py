import pandas as pd
import os

def load_data(filepath='Data/diabetes.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded successfully!")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")
    return df

def quick_summary(df):
    print("\n--- MISSING VALUES ---")
    print("None!" if df.isnull().sum().sum() == 0 else df.isnull().sum())

    print("\n--- TARGET SPLIT ---")
    counts = df['Diabetes_binary'].value_counts()
    total  = len(df)
    print(f"No Diabetes : {counts[0]:,}  ({counts[0]/total*100:.1f}%)")
    print(f"Diabetes    : {counts[1]:,}  ({counts[1]/total*100:.1f}%)")

    print("\n--- FIRST 3 ROWS ---")
    print(df.head(3))

if __name__ == "__main__":
    df = load_data()
    quick_summary(df)


