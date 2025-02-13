import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "ApplicantIncome": np.random.randint(2000, 10000, 500),
    "CreditHistory": np.random.choice([0,1], size=500, p=[0.2, 0.8]),
    "LoanAmount": np.random.randint(50, 300, 500),
    "Dependents": np.random.choice([0, 1, 2, 3], size=500),
    "LoanApproved": np.random.choice([0, 1], size=500, p=[0.4, 0.6])
}

# Create a data frame
loan_data = pd.DataFrame(data)

# Save to CSV
loan_data.to_csv("loan_approval.csv", index=False)

print("Synthetic dataset saved as loan_approval.csv")