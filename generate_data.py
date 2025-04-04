import numpy as np
import pandas as pd
import os

def get_data(n_samples = 250, seeding = 24004):
    np.random.seed(seeding)
    data = pd.DataFrame({
        'Farmer_ID': [f'F{i:05d}' for i in range(1, n_samples + 1)],
        'Farm_Size_ha': np.round(np.random.gamma(2, 1.5, n_samples), 2),
        'Land_Tenure_Type': np.random.choice(['Owned', 'Leased', 'Mixed'], n_samples, p=[0.65, 0.25, 0.10]),
        'Farmer_Age': np.random.randint(20, 75, n_samples),
        'Education_Level': np.random.choice(['None', 'Primary', 'Secondary', 'Higher'], n_samples, p=[0.20, 0.40, 0.30, 0.10]),
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.75, 0.25]),
        'Potential_Carbon_Credits_tCO2e': np.round(np.random.gamma(2.5, 1.8, n_samples), 2),
        'Standalone_Payoff_INR': np.round(np.random.normal(20000, 5000, n_samples), -2),
        'True_Cost_per_Credit_INR': np.round(np.random.gamma(3, 800, n_samples) + 500, -2),
        'Certification_Cost_Individual_INR': np.round(np.random.uniform(4000, 9000, n_samples), -2),
        'Debt_Status': np.random.choice(['Yes', 'No'], n_samples, p=[0.40, 0.60]),
        'Water_Usage_m3': np.round(np.random.gamma(2, 1500, n_samples), 2),
        'Soil_Organic_Carbon_t_per_ha': np.round(np.random.uniform(0.5, 3.0, n_samples), 2),
        'Crop_Type': np.random.choice(['Rice', 'Maize', 'Cotton', 'Sugarcane', 'Wheat', 'Millets'], n_samples, p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10]),
        'Crop_Yield_t_per_ha': np.round(np.random.normal(4.5, 1.5, n_samples), 2),
        'Market_Access': np.random.choice(['Easy', 'Moderate', 'Difficult'], n_samples, p=[0.30, 0.50, 0.20]),
        'Economic_Vulnerability_Score': np.round(np.random.beta(2, 5, n_samples)*100, 2),
        'Previous_Coalition_Experience': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'Farm_Location_State': np.random.choice(['Karnataka', 'Tamil Nadu', 'Kerala', 'Andhra Pradesh', 'Telangana'], n_samples, p=[0.20, 0.25, 0.15, 0.20, 0.20]),
        'Farm_Location_District': np.random.randint(1, 50, n_samples),
        'Risk_Aversion_Coeff': np.round(np.random.beta(2, 3, n_samples) * 5 + 0.1, 2)
    })
    
    data['True_Cost_per_Credit_INR'] = data['True_Cost_per_Credit_INR'].clip(lower=100)
    data['Standalone_Payoff_INR'] = data['Standalone_Payoff_INR'].clip(lower=5000) 
    data['Potential_Carbon_Credits_tCO2e'] = data['Potential_Carbon_Credits_tCO2e'].clip(lower=0.1)

    return data

if __name__ == '__main__':
    n_samples = 250
    output_dir = './data/synthetic/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'indian_farmers_carbon_market_{n_samples}.csv')

    data = get_data(n_samples)
    data.to_csv(output_file, index=False)
    print(f'[+] Saved {n_samples} samples at {output_file}')
    
