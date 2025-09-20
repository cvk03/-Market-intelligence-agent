import pandas as pd
from typing import List, Dict
import os
import json

def prepare_vector_data(rate_df: pd.DataFrame, claims_df: pd.DataFrame, regulatory_data: List[Dict] = None) -> List[str]:
    """Prepare data for vector indexing"""
    texts = []
    
    # Process rate data - group by provider and state
    print("Processing rate data...")
    rate_summary = rate_df.groupby(['provider', 'state', 'insurance_type']).agg({
        'monthly_rate': ['mean', 'min', 'max', 'count'],
        'deductible': 'mean',
        'coverage_amount': 'mean'
    }).round(2)
    
    # Convert rate summaries to text
    for idx, row in rate_summary.iterrows():
        provider, state, ins_type = idx
        text = f"{provider} offers {ins_type} insurance in {state} "
        text += f"with average monthly rate ${row[('monthly_rate', 'mean')]}, "
        text += f"ranging from ${row[('monthly_rate', 'min')]} to ${row[('monthly_rate', 'max')]}. "
        text += f"Based on {int(row[('monthly_rate', 'count')])} policies. "
        text += f"Average deductible is ${row[('deductible', 'mean')]}, "
        text += f"average coverage amount is ${row[('coverage_amount', 'mean')]}."
        texts.append(text)
    
    # Process claims data
    print("Processing claims data...")
    claims_summary = claims_df.groupby(['provider', 'state', 'insurance_type']).agg({
        'claim_amount': ['mean', 'count', 'sum', 'max'],
        'settlement_days': 'mean'
    }).round(2)
    
    # Convert claims summaries to text
    for idx, row in claims_summary.iterrows():
        provider, state, ins_type = idx
        text = f"{provider} in {state} for {ins_type} insurance has processed "
        text += f"{int(row[('claim_amount', 'count')])} claims with average amount ${row[('claim_amount', 'mean')]}. "
        text += f"Total claims value: ${row[('claim_amount', 'sum')]}, "
        text += f"largest claim: ${row[('claim_amount', 'max')]}. "
        text += f"Average settlement time: {row[('settlement_days', 'mean')]} days."
        texts.append(text)
    
    # Add market overview texts
    print("Creating market overview...")
    for state in rate_df['state'].unique():
        state_data = rate_df[rate_df['state'] == state]
        for ins_type in state_data['insurance_type'].unique():
            type_data = state_data[state_data['insurance_type'] == ins_type]
            avg_rate = type_data['monthly_rate'].mean()
            provider_count = type_data['provider'].nunique()
            min_rate = type_data['monthly_rate'].min()
            max_rate = type_data['monthly_rate'].max()
            
            text = f"In {state}, {ins_type} insurance market has {provider_count} providers "
            text += f"with average monthly rate of ${avg_rate:.2f}. "
            text += f"Rates range from ${min_rate} to ${max_rate}."
            texts.append(text)
    
    # Add regulatory data if provided
    if regulatory_data:
        print("Processing regulatory data...")
        for filing in regulatory_data:
            text = f"Regulatory filing {filing['filing_id']} in {filing['state']}: "
            text += f"{filing['description']} "
            text += f"Filed on {filing['filing_date']}, effective {filing['effective_date']}. "
            text += f"Impact: {filing['impact']}. "
            if filing['provider'] != 'Multiple':
                text += f"Filed by {filing['provider']}."
            texts.append(text)
    
    print(f"✅ Prepared {len(texts)} text documents for indexing")
    return texts

if __name__ == "__main__":
    # Load data
    rate_df = pd.read_csv('data/sample_rates.csv')
    claims_df = pd.read_csv('data/claims_data.csv')
    
    # Load regulatory data if exists
    regulatory_data = []
    if os.path.exists('data/regulatory_filings.json'):
        with open('data/regulatory_filings.json', 'r') as f:
            regulatory_data = json.load(f)
    
    # Prepare texts
    texts = prepare_vector_data(rate_df, claims_df, regulatory_data)
    
    # Show sample
    print("\nSample prepared texts:")
    for i, text in enumerate(texts[:3]):
        print(f"\n{i+1}. {text}")