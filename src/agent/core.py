import os
from typing import Dict, Any
import semantic_kernel as sk
from semantic_kernel.functions import kernel_function
from dotenv import load_dotenv
from src.agent.gemini_service import GeminiService

load_dotenv()

class MarketIntelligenceAgent:
    def __init__(self):
        """Initialize the agent with Semantic Kernel and Gemini"""
        print("Initializing Market Intelligence Agent...")

        # Create kernel
        self.kernel = sk.Kernel()

        # Initialize Gemini service
        self.gemini_service = GeminiService()

        # Register skills
        self._register_skills()

        print("✅ Agent initialized successfully!")

    def _register_skills(self):
        """Register agent skills"""
        from semantic_kernel.functions import KernelPlugin

        # Create skills plugin
        self.skills = KernelPlugin(
            name="MarketIntelligenceSkills",
            description="Skills for insurance market analysis"
        )

        # Rate benchmarking skill (concise, flexible)
        @kernel_function(
            name="benchmark_rates",
            description="Compare insurance rates across providers"
        )
        def benchmark_rates(
            data: str,
            insurance_type: str = "auto",
            region: str = "CA"
        ) -> str:
            ins_type_txt = insurance_type if insurance_type.lower() != "all" else "all insurance types"
            region_txt = region if region.lower() != "all" else "all states"
            prompt = f"""
You are an expert insurance market analyst.

Insurance Type: {ins_type_txt}
Region: {region_txt}

Data:
{data}

Answer the user's query with a concise, easy-to-understand summary. 
Include key stats, top providers and rates if relevant.
If the data spans multiple regions or products, mention any notable trends or differences.
Avoid rigid bullet formatting unless listing providers/rates is most relevant.

Example:
"The average auto insurance premium in California is $195. The three cheapest providers are Progressive ($185), Geico ($190), and StateFarm ($200). Progressive's rate is currently the lowest in the market."
"""
            return prompt

        # Trend analysis skill (concise, flexible)
        @kernel_function(
            name="analyze_trends",
            description="Analyze insurance market trends"
        )
        def analyze_trends(
            data: str,
            time_period: str = "12 months"
        ) -> str:
            prompt = f"""
You are analyzing insurance market trends.

Historical Data:
{data}

Summarize key claim or premium trends for the past {time_period} in a concise and business-friendly manner.
Highlight only the most significant changes, actionable insights, or shifts.
Use percentages or numbers where possible, and keep each point clear and brief.

Example:
"Claims volume rose 12% in the last 12 months, with a spike in Q2. Average premiums increased by 7%. Progressive gained 2% market share while losses rose for small providers."
"""
            return prompt

        # Recommendation skill (concise, natural)
        @kernel_function(
            name="generate_recommendations",
            description="Generate pricing strategy recommendations"
        )
        def generate_recommendations(
            market_data: str,
            current_position: str = "mid-market"
        ) -> str:
            prompt = f"""
You are a pricing strategy advisor for insurance.

Market Data:
{market_data}

Give 3-5 actionable recommendations for pricing strategy, each as a short and clear statement.
No lengthy rationale, just the action and a brief reason. Use numbers/targets if possible.

Example:
- Lower rates 5% for low-risk zip codes to boost competitiveness.
- Increase premiums for high claim segments by 8% to improve profitability.
- Add young driver discounts to expand market share.
"""
            return prompt

        # Add functions to skills
        self.skills.add([benchmark_rates, analyze_trends, generate_recommendations])

        # Add plugin to kernel
        self.kernel.add_plugin(self.skills)

        print("✅ Skills registered successfully!")

    async def process_query(
        self, 
        query: str, 
        retrieved_data: str,
        insurance_type: str = "auto",
        region: str = "CA"
    ) -> Dict[str, Any]:
        """Process user query and return analysis"""
        try:
            # Skill selection based on query keywords.
            query_lower = query.lower()
            if any(word in query_lower for word in [
                "benchmark", "compare", "rate", "cheapest", "lowest", "provider", "price", "premium"
            ]):
                skill_name = "benchmark_rates"
                arguments = {
                    "data": retrieved_data,
                    "insurance_type": insurance_type,
                    "region": region
                }
            elif any(word in query_lower for word in ["trend", "pattern", "claim", "change", "increase", "decrease"]):
                skill_name = "analyze_trends"
                arguments = {
                    "data": retrieved_data,
                    "time_period": "12 months"
                }
            else:
                skill_name = "generate_recommendations"
                arguments = {
                    "market_data": retrieved_data,
                    "current_position": "mid-market"
                }

            print(f"🔄 Processing query with skill: {skill_name}")

            # Get the function
            function = self.kernel.get_function("MarketIntelligenceSkills", skill_name)

            # Create the prompt using the function
            prompt_result = await self.kernel.invoke(function, **arguments)
            prompt = str(prompt_result)

            # Get response from Gemini
            response = await self.gemini_service.generate_response(prompt)

            return {
                "success": True,
                "query_type": skill_name,
                "analysis": response,
                "confidence": 0.9
            }

        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis": "Sorry, I encountered an error processing your query."
            }

    def process_query_sync(
        self, 
        query: str, 
        retrieved_data: str,
        insurance_type: str = "auto",
        region: str = "CA"
    ) -> Dict[str, Any]:
        """Synchronous version of process_query for compatibility"""
        import asyncio
        return asyncio.run(self.process_query(query, retrieved_data, insurance_type, region))