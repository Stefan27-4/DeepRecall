"""
Example: Basic memory recall with DeepRecall

This example shows how to use DeepRecall to query an AI agent's memory files.
"""

import sys
sys.path.insert(0, "../skill")

from deep_recall import recall, recall_quick, recall_deep


# Example 1: Quick identity check (fastest, cheapest)
print("=" * 60)
print("Example 1: Quick identity recall")
print("=" * 60)
result = recall_quick("What is my human's name and what timezone are they in?")
print(result)


# Example 2: Standard memory query
print("\n" + "=" * 60)
print("Example 2: Standard memory recall")
print("=" * 60)
result = recall("What projects have we worked on together?")
print(result)


# Example 3: Deep comprehensive search
print("\n" + "=" * 60)
print("Example 3: Deep recall across all files")
print("=" * 60)
result = recall_deep("Summarize all important decisions and milestones")
print(result)


# Example 4: Custom configuration
print("\n" + "=" * 60)
print("Example 4: Custom config recall")
print("=" * 60)
result = recall(
    "What security rules do I need to follow?",
    scope="identity",
    verbose=True,
    config_overrides={
        "max_depth": 1,           # Shallow search
        "max_money_spent": 0.05,  # Very cheap
    }
)
print(result)
