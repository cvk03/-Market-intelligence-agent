import pandas as pd
import json
import os
from typing import List, Dict
import random
from datetime import datetime, timedelta

class SampleDataGenerator:
    def __init__(self):
        self.providers = [
            "StateFarm", "Geico", "Progressive", "Allstate", "USAA",
            "Liberty Mutual", "Farmers", "Nationwide", "Travelers", "AmFam"
        ]
        self.states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
        self.insurance_types = ["auto", "home", "life", "health"]
        
    def generate_rate_data(self, num_records: int = 500) -> pd.DataFrame:
        """Generate sample insurance rate data"""
        print("Generating sample rate data...")
        data = []
        
        for _ in range(num_records):
            record = {
                'provider': random.choice(self.providers),
                'state': random.choice(self.states),
                'insurance_type': random.choice(self.insurance_types),
                'coverage_level': random.choice(['Basic', 'Standard', 'Premium']),
                'monthly_rate': round(random.uniform(80, 400), 2),
                'deductible': random.choice([250, 500, 1000, 2500]),
                'coverage_amount': random.choice([50000, 100000, 250000, 500000]),
                'customer_age_group': random.choice(['18-25', '26-35', '36-50', '51-65', '65+']),
                'rating_factor': round(random.uniform(0.8, 1.5), 2),
                'last_updated': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_claims_data(self, num_records: int = 200) -> pd.DataFrame:
        """Generate sample claims data"""
        print("Generating sample claims data...")
        data = []
        
        for _ in range(num_records):
            claim_date = datetime.now() - timedelta(days=random.randint(1, 365*2))
            record = {
                'claim_id': f"CLM{random.randint(100000, 999999)}",
                'provider': random.choice(self.providers),
                'state': random.choice(self.states),
                'insurance_type': random.choice(self.insurance_types),
                'claim_amount': round(random.uniform(500, 25000), 2),
                'claim_type': random.choice(['Collision', 'Theft', 'Weather', 'Liability', 'Medical']),
                'claim_date': claim_date.strftime('%Y-%m-%d'),
                'settlement_days': random.randint(5, 60),
                'fraud_indicator': random.choice([True, False]) if random.random() < 0.05 else False
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_regulatory_data(self) -> List[Dict]:
        """Generate sample regulatory filing data"""
        print("Generating sample regulatory data...")
        filings = [
            {
                'filing_id': 'REG001',
                'state': 'CA',
                'filing_date': '2025-01-05',
                'effective_date': '2025-02-01',
                'filing_type': 'Rate Change',
                'description': 'Auto insurance rate adjustment - average increase of 3.2%',
                'impact': 'Rate increases for comprehensive coverage in high-risk areas',
                'provider': 'StateFarm'
            },
            {
                'filing_id': 'REG002', 
                'state': 'TX',
                'filing_date': '2025-01-07',
                'effective_date': '2025-03-01',
                'filing_type': 'New Product',
                'description': 'Introduction of usage-based insurance program',
                'impact': 'Potential rate reductions for low-mileage drivers',
                'provider': 'Progressive'
            },
            {
                'filing_id': 'REG003',
                'state': 'FL',
                'filing_date': '2025-01-08',
                'effective_date': '2025-02-15',
                'filing_type': 'Regulatory Change',
                'description': 'Updated hurricane coverage requirements',
                'impact': 'Mandatory coverage changes for coastal properties',
                'provider': 'Multiple'
            }
        ]
        return filings
    
    def save_sample_data(self):
        """Generate and save sample datasets"""
        os.makedirs('data', exist_ok=True)
        
        # Generate data
        rate_data = self.generate_rate_data(500)
        claims_data = self.generate_claims_data(200)
        regulatory_data = self.generate_regulatory_data()
        
        # Save to files
        rate_data.to_csv('data/sample_rates.csv', index=False)
        claims_data.to_csv('data/claims_data.csv', index=False)
        
        with open('data/regulatory_filings.json', 'w') as f:
            json.dump(regulatory_data, f, indent=2)
        
        print("✅ Sample data generated and saved!")
        print(f"  - Rates: {len(rate_data)} records")
        print(f"  - Claims: {len(claims_data)} records")
        print(f"  - Regulatory: {len(regulatory_data)} filings")
        
        return rate_data, claims_data, regulatory_data

if __name__ == "__main__":
    generator = SampleDataGenerator()
    generator.save_sample_data()
