import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import SampleDataGenerator
from src.utils.vector_store import SimpleVectorStore
from src.utils.prepare_data import prepare_vector_data
from src.agent.core import MarketIntelligenceAgent
import pandas as pd
import json

async def test_agent():
    """Test the complete agent pipeline"""
    print("="*60)
    print("🚀 Market Intelligence Agent Setup with Google Gemini")
    print("="*60)
    
    # Step 1: Generate sample data
    print("\n📊 Step 1: Generating sample data...")
    generator = SampleDataGenerator()
    rate_data, claims_data, regulatory_data = generator.save_sample_data()
    
    # Step 2: Prepare text data
    print("\n📝 Step 2: Preparing data for vector store...")
    texts = prepare_vector_data(rate_data, claims_data, regulatory_data)
    
    # Step 3: Create vector store
    print("\n🔍 Step 3: Creating vector store...")
    vector_store = SimpleVectorStore()
    vector_store.create_index(texts)
    vector_store.save()
    
    # Step 4: Initialize agent
    print("\n🤖 Step 4: Initializing agent with Gemini...")
    try:
        agent = MarketIntelligenceAgent()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Please make sure you have:")
        print("1. Created a .env file in the project root")
        print("2. Added your Google API key: GOOGLE_API_KEY=your_key_here")
        print("3. Get your free API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Step 5: Test queries
    print("\n✨ Step 5: Testing agent with sample queries...")
    print("="*60)
    
    test_queries = [
        ("Compare auto insurance rates in California", "auto", "CA")
    ]
    
    for query, ins_type, region in test_queries:
        print(f"\n📌 Query: {query}")
        print(f"   Insurance Type: {ins_type}, Region: {region}")
        print("-"*60)
        
        # Search for relevant data
        search_results = vector_store.search(query, k=10)
        relevant_data = "\n".join([r['text'] for r in search_results[:5]])
        
        print(f"Found {len(search_results)} relevant documents")
        
        # Process with agent
        result = await agent.process_query(
            query=query,
            retrieved_data=relevant_data,
            insurance_type=ins_type,
            region=region
        )
        
        if result['success']:
            print(f"✅ Analysis Type: {result['query_type']}")
            print(f"\n📊 Analysis Preview:\n{result['analysis'][:500]}...")
            print(f"\n[Full analysis contains {len(result['analysis'])} characters]")
        else:
            print(f"❌ Error: {result['error']}")
    
    print("\n"+"="*60)
    print("✅ Agent setup and testing complete!")
    print("="*60)
    
    # Save a summary
    with open("setup_summary.txt", "w") as f:
        f.write("Market Intelligence Agent Setup Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Data Generated:\n")
        f.write(f"- Rate records: {len(rate_data)}\n")
        f.write(f"- Claims records: {len(claims_data)}\n")
        f.write(f"- Regulatory filings: {len(regulatory_data)}\n")
        f.write(f"- Vector store documents: {len(texts)}\n")
        f.write(f"\nAgent ready for use!\n")

if __name__ == "__main__":
    print("\n🌟 Welcome to Market Intelligence Agent Setup!")
    print("\n⚠️  Prerequisites:")
    print("1. Make sure you have created a .env file with your Google API key")
    print("2. Get your free API key from: https://makersuite.google.com/app/apikey")
    print("3. Add to .env: GOOGLE_API_KEY=your_key_here")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ ERROR: GOOGLE_API_KEY not found in environment!")
        print("Please set up your .env file first.")
        sys.exit(1)
    else:
        print(f"\n✅ API Key found: {api_key[:10]}...")
    
    input("\nPress Enter to start the setup...")
    
    # Run the test
    asyncio.run(test_agent())