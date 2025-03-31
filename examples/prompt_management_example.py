"""
Example of using the prompt management system.
Shows template creation, versioning, A/B testing, and optimization.
"""
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path so we can import from there
src_path = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, src_path)

from src.prompts import PromptManager, PromptTemplate
from src.utils.logger import LoggerFactory

# Set up storage directory for prompt data
PROMPT_STORAGE_DIR = os.path.join(src_path, "data", "prompts")
os.makedirs(PROMPT_STORAGE_DIR, exist_ok=True)


def basic_example():
    """Example of basic prompt templating."""
    print("\n=== Basic Prompt Templating Example ===")
    
    # Create a prompt template directly
    template = PromptTemplate(
        name="Basic Greeting",
        description="A simple greeting template",
        template="Hello, {{name}}! Welcome to {{service}}.",
        default_values={"service": "our service"}
    )
    
    # Render with different variables
    result1 = template.render({"name": "Alice"})
    result2 = template.render({"name": "Bob", "service": "the Prompt Management System"})
    
    print(f"Template: {template.template}")
    print(f"Rendered with default service: {result1}")
    print(f"Rendered with custom service: {result2}")


def prompt_manager_example():
    """Example of using the PromptManager."""
    print("\n=== Prompt Manager Example ===")
    
    # Create prompt manager
    manager = PromptManager(
        storage_dir=PROMPT_STORAGE_DIR,
        logger=LoggerFactory.create()
    )
    
    # Create a template
    template_id = manager.create_template(
        name="Query Generator",
        description="Generates a query for a search engine",
        template="Create a search query for {{search_engine}} to find information about {{topic}}.",
        default_values={"search_engine": "Google"}
    )
    
    print(f"Created template with ID: {template_id}")
    
    # Create a new version
    version_id = manager.create_version(
        template_id=template_id,
        template_string="Generate an optimized search query for {{search_engine}} to find detailed information about {{topic}}. Focus on {{aspect}} aspects.",
        default_values={"search_engine": "Google", "aspect": "technical"},
        name="Enhanced Version",
        description="Added aspect parameter for more focused queries",
        set_active=True
    )
    
    print(f"Created new version with ID: {version_id}")
    
    # List versions
    versions = manager.get_versions(template_id)
    print(f"Template has {len(versions)} versions:")
    for version in versions:
        print(f"  - Version {version['version']}: {version['name']} (Active: {version['is_active']})")
    
    # Render prompt
    prompt, usage_id = manager.render_prompt(
        template_id=template_id,
        variables={"topic": "machine learning algorithms", "aspect": "performance"},
        user_id="example-user-123"
    )
    
    print(f"Rendered prompt: {prompt}")
    
    # Record performance metrics
    manager.record_prompt_performance(
        usage_id=usage_id,
        metrics={
            "tokens": 45,
            "latency": 0.23,
            "relevance_score": 0.92
        },
        feedback={"user_rating": 4}
    )
    
    print("Recorded performance metrics")


def ab_testing_example():
    """Example of A/B testing different prompt versions."""
    print("\n=== A/B Testing Example ===")
    
    # Create prompt manager
    manager = PromptManager(
        storage_dir=PROMPT_STORAGE_DIR,
        logger=LoggerFactory.create()
    )
    
    # Create a template for code generation
    template_id = manager.create_template(
        name="Code Generator",
        description="Generates code snippets",
        template="Write a {{language}} function to {{task}}.",
        default_values={"language": "Python"}
    )
    
    # Create a few versions for testing
    version_ids = []
    
    # Version 1 - Basic
    v1_id = manager.create_version(
        template_id=template_id,
        template_string="Write a {{language}} function to {{task}}.",
        name="Basic Version",
        description="Simple instruction",
        set_active=True
    )
    version_ids.append(v1_id)
    
    # Version 2 - Detailed
    v2_id = manager.create_version(
        template_id=template_id,
        template_string="Write a clean, efficient {{language}} function to {{task}}. Include error handling and comments.",
        name="Detailed Version",
        description="More detailed instruction with quality guidelines"
    )
    version_ids.append(v2_id)
    
    # Version 3 - Contextual
    v3_id = manager.create_version(
        template_id=template_id,
        template_string="Given the context of {{context}}, write a {{language}} function to {{task}}. Optimize for {{optimization_goal}}.",
        default_values={"language": "Python", "context": "a web application", "optimization_goal": "readability"},
        name="Contextual Version",
        description="Added context and optimization parameters"
    )
    version_ids.append(v3_id)
    
    print(f"Created template with ID: {template_id} and 3 versions")
    
    # Start A/B test
    manager.start_ab_test(
        template_id=template_id,
        version_ids=version_ids,
        weights=[0.3, 0.3, 0.4]  # 30%, 30%, 40% distribution
    )
    
    print("Started A/B test with custom weights")
    
    # Simulate users and requests
    users = [f"user-{i}" for i in range(1, 11)]
    tasks = [
        "sort an array",
        "parse JSON",
        "connect to a database",
        "implement a binary search",
        "generate random numbers"
    ]
    languages = ["Python", "JavaScript", "Java", "C++"]
    
    print("Simulating 20 users interacting with the template...")
    
    # Record simulated interactions and random performance
    import random
    
    for _ in range(20):
        user_id = random.choice(users)
        task = random.choice(tasks)
        language = random.choice(languages)
        
        # Render prompt (A/B testing happens here)
        prompt, usage_id = manager.render_prompt(
            template_id=template_id,
            variables={
                "task": task,
                "language": language,
                "context": "a production environment",
                "optimization_goal": random.choice(["speed", "memory", "readability"])
            },
            user_id=user_id
        )
        
        # Simulate performance metrics
        manager.record_prompt_performance(
            usage_id=usage_id,
            metrics={
                "tokens": random.randint(30, 120),
                "latency": random.uniform(0.1, 1.5),
                "quality_score": random.uniform(0.5, 1.0)
            },
            feedback={
                "user_rating": random.randint(1, 5)
            }
        )
        
        # Small delay
        time.sleep(0.1)
    
    # Compare version performance
    comparison = manager.compare_versions(
        template_id=template_id,
        version_ids=version_ids,
        metric_keys=["quality_score", "user_rating"]
    )
    
    print("\nVersion comparison results:")
    for version_id, metrics in comparison.get("versions", {}).items():
        version_name = next((v["name"] for v in manager.get_versions(template_id) if v["version_id"] == version_id), "Unknown")
        print(f"\n  {version_name}:")
        print(f"    Usage count: {metrics.get('usage_count')}")
        
        for metric_name, metric_values in metrics.get("metrics", {}).items():
            print(f"    {metric_name}: avg={metric_values.get('avg', 0):.2f}, min={metric_values.get('min', 0):.2f}, max={metric_values.get('max', 0):.2f}")
    
    # Find best version based on quality score
    best_version = manager.optimize_prompt(
        template_id=template_id,
        metric_key="quality_score",
        higher_is_better=True,
        min_usage_count=3
    )
    
    if best_version:
        best_version_name = next((v["name"] for v in manager.get_versions(template_id) if v["version_id"] == best_version), "Unknown")
        print(f"\nBest performing version: {best_version_name}")
    else:
        print("\nNot enough data to determine best version")
    
    # Stop A/B test with winning version
    manager.stop_ab_test(template_id, best_version)
    print("Stopped A/B test and set winning version as active")


def main():
    """Run examples of the prompt management system."""
    # Run basic examples
    basic_example()
    
    # Run PromptManager examples
    prompt_manager_example()
    
    # Run A/B testing examples
    ab_testing_example()


if __name__ == "__main__":
    main() 