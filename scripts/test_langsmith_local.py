import os
import sys
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.enrich_inference import generate_inference_description

def test_monitoring():
    print("--- LangSmith Local Test ---")
    
    # Check for required environment variables
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT")
    
    if not api_key:
        print("❌ Error: LANGCHAIN_API_KEY is not set.")
        return
    
    print(f"✅ LANGCHAIN_API_KEY found (ends in ...{api_key[-4:]})")
    print(f"✅ LANGCHAIN_PROJECT is set to: {project or 'default'}")
    
    # Force tracing for this test
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    # Sample input data
    sample_data = {
        "Age": 33,
        "Sex": "male",
        "Job": 2,
        "Housing": "own",
        "Saving accounts": "little",
        "Checking account": "moderate",
        "Credit amount": 2500,
        "Duration": 12,
        "Purpose": "car"
    }

    print("\nExecuting generate_inference_description...")
    try:
        # This function is decorated with @traceable
        description = generate_inference_description(sample_data)
        print("\n✨ Result from LLM:")
        print(description)
        print("\n✅ Execution completed. Check your LangSmith dashboard now!")
        print(f"Link: https://smith.langchain.com/o/default/projects/p/{project if project else 'default'}?tab=traces")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")

if __name__ == "__main__":
    test_monitoring()
