import pandas as pd

def main():
    # Read the original Monte Carlo file
    df = pd.read_csv("Probability.csv")

    # Subtract 5 from every numeric value with a minimum of 17
    df_shifted = df.applymap(
        lambda x: max(x - 5, 17) if isinstance(x, (int, float)) else x
    )

    # Output as offset.csv
    df_shifted.to_csv("Monte_Carlo.csv", index=False)

if __name__ == "__main__":
    main()
