import requests
import json

BASE_URL = "http://localhost:8000"

def test_query(question):
    """Test a single query"""
    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": question}
    )
    result = response.json()
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(f"Answer: {result['answer']}")
    print(f"\nSources Used:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n[{i}] {source[:200]}...")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Test 3 questions
    questions = [
        "What is this document about?",
        "What are the key points mentioned?",
        "Can you summarize the main topics?"
    ]
    
    for q in questions:
        test_query(q)

