import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the model server type
model_server = os.getenv('MODEL_SERVER', 'GROQ').upper()

# Configure API credentials based on model server
if model_server == "GROQ":
    API_KEY = os.getenv('GROQ_API_KEY')
    BASE_URL = os.getenv('GROQ_BASE_URL')
    LLM_MODEL = os.getenv('GROQ_MODEL')
elif model_server == "NGU":
    API_KEY = os.getenv('NGU_API_KEY')
    BASE_URL = os.getenv('NGU_BASE_URL')
    LLM_MODEL = os.getenv('NGU_MODEL')
else:
    raise ValueError(f"Unsupported MODEL_SERVER: {model_server}")

# Initialize the OpenAI client with custom base URL
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Define tool schemas
def get_tool_schemas():
    return {
        "extract_key_points": {
            "type": "function",
            "function": {
                "name": "extract_key_points",
                "description": "Extract key points from a blog post",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the blog post"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content of the blog post"
                        },
                        "key_points": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of key points extracted from the blog post"
                        }
                    },
                    "required": ["key_points"]
                }
            }
        },
        "generate_summary": {
            "type": "function",
            "function": {
                "name": "generate_summary",
                "description": "Generate a concise summary from the key points",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Concise summary of the blog post"
                        }
                    },
                    "required": ["summary"]
                }
            }
        },
        "create_social_media_posts": {
            "type": "function",
            "function": {
                "name": "create_social_media_posts",
                "description": "Create social media posts for different platforms",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "twitter": {
                            "type": "string",
                            "description": "Post optimized for Twitter/X (max 280 characters)"
                        },
                        "linkedin": {
                            "type": "string",
                            "description": "Post optimized for LinkedIn (professional tone)"
                        },
                        "facebook": {
                            "type": "string",
                            "description": "Post optimized for Facebook"
                        }
                    },
                    "required": ["twitter", "linkedin", "facebook"]
                }
            }
        },
        "create_email_newsletter": {
            "type": "function",
            "function": {
                "name": "create_email_newsletter",
                "description": "Create an email newsletter from the blog post and summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Email subject line"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body content in plain text"
                        }
                    },
                    "required": ["subject", "body"]
                }
            }
        }
    }

# Function to make LLM API calls
def call_llm(messages, tools=None, tool_choice=None):
    try:
        kwargs = {
            "model": LLM_MODEL,
            "messages": messages
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

# Function to read sample blog post
def get_sample_blog_post():
    try:
        with open('sample_blog_post.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: sample_blog_post.json file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in sample_blog_post.json.")
        return None

# Task Functions with Tool Calling
def task_extract_key_points(blog_post):
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles."},
        {"role": "user", "content": f"Extract the key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    
    tool_schemas = get_tool_schemas()
    response = call_llm(
        messages=messages,
        tools=[tool_schemas["extract_key_points"]],
        tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
    )
    
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("key_points", [])
    return []

def task_generate_summary(key_points, max_length=150):
    messages = [
        {"role": "system", "content": "You are an expert at summarizing content concisely while preserving key information."},
        {"role": "user", "content": f"Generate a summary based on these key points, max {max_length} words:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    
    tool_schemas = get_tool_schemas()
    response = call_llm(
        messages=messages,
        tools=[tool_schemas["generate_summary"]],
        tool_choice={"type": "function", "function": {"name": "generate_summary"}}
    )
    
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("summary", "")
    return ""

def task_create_social_media_posts(key_points, blog_title):
    messages = [
        {"role": "system", "content": "You are a social media expert who creates engaging posts optimized for different platforms."},
        {"role": "user", "content": f"Create social media posts for Twitter, LinkedIn, and Facebook based on this blog title: '{blog_title}' and these key points:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    
    tool_schemas = get_tool_schemas()
    response = call_llm(
        messages=messages,
        tools=[tool_schemas["create_social_media_posts"]],
        tool_choice={"type": "function", "function": {"name": "create_social_media_posts"}}
    )
    
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"twitter": "", "linkedin": "", "facebook": ""}

def task_create_email_newsletter(blog_post, summary, key_points):
    messages = [
        {"role": "system", "content": "You are an email marketing specialist who creates engaging newsletters."},
        {"role": "user", "content": f"Create an email newsletter based on this blog post:\n\nTitle: {blog_post['title']}\n\nSummary: {summary}\n\nKey Points:\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    
    tool_schemas = get_tool_schemas()
    response = call_llm(
        messages=messages,
        tools=[tool_schemas["create_email_newsletter"]],
        tool_choice={"type": "function", "function": {"name": "create_email_newsletter"}}
    )
    
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"subject": "", "body": ""}

# Pipeline Workflow
def run_pipeline_workflow(blog_post):
    # Extract key points
    print("Extracting key points...")
    key_points = task_extract_key_points(blog_post)
    
    # Generate summary from key points
    print("Generating summary...")
    summary = task_generate_summary(key_points)
    
    # Create social media posts
    print("Creating social media posts...")
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    
    # Create email newsletter
    print("Creating email newsletter...")
    email = task_create_email_newsletter(blog_post, summary, key_points)
    
    # Return all generated content
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

# DAG Workflow
def run_dag_workflow(blog_post):
    # Start with extracting key points (this is still the first step)
    print("Extracting key points...")
    key_points = task_extract_key_points(blog_post)
    
    # Generate summary from key points
    print("Generating summary...")
    summary = task_generate_summary(key_points)
    
    # These two tasks can run in parallel since they don't depend on each other
    # In a real DAG system, you'd use async or threading to run these concurrently
    
    # Create social media posts (depends only on key_points)
    print("Creating social media posts...")
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    
    # Create email newsletter (depends on blog_post, summary, and key_points)
    print("Creating email newsletter...")
    email = task_create_email_newsletter(blog_post, summary, key_points)
    
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

def extract_key_points_with_cot(blog_post):
    """
    Extract key points from a blog post using chain-of-thought reasoning.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        List of key points
    """
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles."},
        {"role": "user", "content": f"I want you to extract the key points from this blog post. Before giving me the final list, think step-by-step about:\n\n1. What are the main themes of the article?\n2. What are the most important claims or arguments?\n3. What evidence or examples support these claims?\n4. What are the practical implications or takeaways?\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    
    # First, get the model to think through the content
    thinking_response = call_llm(messages=messages)
    
    if thinking_response:
        thinking = thinking_response.choices[0].message.content
        
        # Now ask it to extract the key points in a structured format
        messages.append({"role": "assistant", "content": thinking})
        messages.append({"role": "user", "content": "Based on your analysis, extract the key points as a structured list."})
        
        tool_schemas = get_tool_schemas()
        # Use tool_choice to ensure the model calls our function
        response = call_llm(
            messages=messages,
            tools=[tool_schemas["extract_key_points_schema"]],
            tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
        )
        
        # Extract the tool call information
        if response and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            return result.get("key_points", [])
    
    # Fallback to the regular method if CoT fails
    return task_extract_key_points(blog_post)

# Reflexion and Self-Correction
def evaluate_content(content, content_type):
    messages = [
        {"role": "system", "content": "You are an expert content evaluator who provides detailed and constructive feedback."},
        {"role": "user", "content": f"Evaluate the quality of this {content_type}:\n\n{content}\n\nProvide a quality score between 0 and 1 and specific feedback for improvement."}
    ]
    
    response = call_llm(messages)
    if response:
        evaluation_text = response.choices[0].message.content
        
        # Extract quality score and feedback
        score_match = re.search(r'Quality Score: (\d+(\.\d+)?)', evaluation_text)
        quality_score = float(score_match.group(1)) if score_match else 0.5
        
        return {
            "quality_score": quality_score,
            "feedback": evaluation_text
        }
    return {"quality_score": 0.5, "feedback": "No detailed feedback available."}


def improve_content(content, feedback, content_type):
    """
    Improve content based on feedback.
    
    Args:
        content: The content to improve (string, list, or dictionary)
        feedback: Specific feedback on how to improve the content
        content_type: The type of content (e.g., "summary", "social_media_post", "email", "key_points")
    
    Returns:
        Improved content in the same format as the input
    """
    # Format the content appropriately based on type
    content_str = content
    if isinstance(content, list):
        content_str = "\n".join([f"- {item}" for item in content])
    elif isinstance(content, dict):
        content_str = "\n\n".join([f"{key.upper()}:\n{value}" for key, value in content.items()])
    

    tool_schemas = get_tool_schemas()

    # Create messages for LLM
    messages = [
        {"role": "system", "content": "You are a content improvement specialist who makes targeted enhancements based on feedback."},
        {"role": "user", "content": f"Improve the following {content_type} based on this feedback:\n\nFeedback: {feedback}\n\nOriginal content:\n\n{content_str}"}
    ]
    
    # Define schema and tool name based on content type
    if content_type == "summary":
        schema = tool_schemas["generate_summary"]
        tool_name = "generate_summary"
    elif content_type == "social_media_post":
        schema = tool_schemas["create_social_media_posts"]
        tool_name = "create_social_media_posts"
    elif content_type == "email":
        schema = tool_schemas["create_email_newsletter"]
        tool_name = "create_email_newsletter"
    elif content_type == "key_points":
        schema = tool_schemas["extract_key_points"]
        tool_name = "extract_key_points"
    else:
        # Generic improvement schema for unknown content types
        schema = {
            "type": "function",
            "function": {
                "name": "improve_content",
                "description": f"Improve {content_type}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "improved_content": {
                            "type": "string",
                            "description": "Improved content"
                        }
                    },
                    "required": ["improved_content"]
                }
            }
        }
        tool_name = "improve_content"
    
    # Call LLM and process response
    try:
        response = call_llm(
            messages=messages,
            tools=[schema],
            tool_choice={"type": "function", "function": {"name": tool_name}}
        )
        
        if response and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)

            # Handle different response formats based on content type
            if content_type == "summary":
                return result.get("summary", content)
            elif content_type == "social_media_post":
                return result
            elif content_type == "email":
                return result
            elif content_type == "key_points":
                return result.get("key_points", content)
            else:
                return result.get("improved_content", content)
            
    except Exception as e:
        print(f"Error improving content: {e}")
    
    # Return original content if anything fails
    return content

def generate_with_reflexion(generator_func, max_attempts=3):
    """
    Apply Reflexion to a content generation function.
    Args:
        generator_func: Function that generates content
        max_attempts: Maximum number of correction attempts
    Returns:
        Function that generates self-corrected content
    """
    def wrapped_generator(*args, **kwargs):
        # Get the content type from kwargs or use a default
        content_type = kwargs.pop("content_type", "content")
        
        # Generate initial content
        content = generator_func(*args, **kwargs)
        
        print(f"Initial {content_type} generated. Evaluating quality...")
        
        # Evaluate and correct if needed
        for attempt in range(max_attempts):
            evaluation = evaluate_content(content, content_type)
            print(f"Quality score: {evaluation['quality_score']}")
            
            # If quality is good enough, return the content
            if evaluation["quality_score"] >= 0.8:  # Threshold for acceptable quality
                print(f"{content_type.capitalize()} is good quality. No further improvements needed.")
                return content
            
            # Otherwise, attempt to improve the content
            print(f"Attempting to improve {content_type} (attempt {attempt+1}/{max_attempts})...")
            print(f"Feedback: {evaluation['feedback']}")
            
            improved_content = improve_content(content, evaluation["feedback"], content_type)
            content = improved_content
        
        print(f"Final {content_type} after {max_attempts} improvement attempts.")
        return content
    
    return wrapped_generator

# Reflexion Workflow
def run_workflow_with_reflexion(blog_post):
    # Apply Reflexion to each task
    # Extract key points with reflexion
    print("Extracting key points with reflexion...")
    key_points = generate_with_reflexion(task_extract_key_points)(blog_post, content_type="key_points")
    
    # Generate summary with reflexion
    print("Generating summary with reflexion...")
    summary = generate_with_reflexion(task_generate_summary)(key_points, content_type="summary")
    
    # Create social media posts with reflexion
    print("Creating social media posts with reflexion...")
    social_posts = generate_with_reflexion(task_create_social_media_posts)(key_points, blog_post['title'], content_type="social_media_post")
    
    # Create email newsletter with reflexion
    print("Creating email newsletter with reflexion...")
    email = generate_with_reflexion(task_create_email_newsletter)(blog_post, summary, key_points, content_type="email")
    
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

# Agent-Driven Workflow

def define_agent_tools():
    """
    Define the tools that the workflow agent can use.
    Returns:
        List of tool definitions
    """
    # Collect all previously defined tools
    all_tools = get_tool_schemas()
    
    # Add a "finish" tool
    finish_tool_schema = {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Complete the workflow and return the final results",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The final summary"
                    },
                    "social_posts": {
                        "type": "object",
                        "properties": {
                            "twitter": {
                                "type": "string",
                                "description": "Post for Twitter/X"
                            },
                            "linkedin": {
                                "type": "string",
                                "description": "Post for LinkedIn"
                            },
                            "facebook": {
                                "type": "string",
                                "description": "Post for Facebook"
                            }
                        },
                        "description": "The social media posts for each platform"
                    },
                    "email": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "Email subject line"
                            },
                            "body": {
                                "type": "string",
                                "description": "Email body content"
                            }
                        },
                        "description": "The email newsletter"
                    }
                },
                "required": ["summary", "social_posts", "email"]
            }
        }
    }
    
    all_tools["finish"] = finish_tool_schema
    # Return all tools, including the finish tool
    return all_tools

def run_agent_workflow(blog_post):
    system_message = """
    You are a Content Repurposing Agent. Your job is to take a blog post and repurpose it into different formats:
    1. Extract key points from the blog post
    2. Generate a concise summary
    3. Create social media posts for different platforms
    4. Create an email newsletter
    You have access to tools that can help you with each of these tasks. Think carefully about which tools to use and in what order.
    When you're done, use the 'finish' tool to complete the workflow.
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please repurpose this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    
    # Define the agent tools
    tool_schemas = define_agent_tools()
    tools = list(tool_schemas.values())
    
    results = {}
    max_iterations = 10
    
    for _ in range(max_iterations):
        response = call_llm(messages, tools)
        messages.append(response.choices[0].message)
        
        if not response.choices[0].message.tool_calls:
            break
        
        for tool_call in response.choices[0].message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if tool_name == "extract_key_points":
                results["key_points"] = arguments.get("key_points", [])
            elif tool_name == "generate_summary":
                results["summary"] = arguments.get("summary", "")
            elif tool_name == "create_social_media_posts":
                results["social_posts"] = arguments
            elif tool_name == "create_email_newsletter":
                results["email_newsletter"] = arguments
    
    return results

# Bonus Challenge: Comparative Evaluation
def comparative_workflow_evaluation(blog_post):
    print("Running Workflow Comparisons...")
    
    workflows = {
        "Pipeline Workflow": run_pipeline_workflow,
        "Reflexion Workflow": run_workflow_with_reflexion,
        "Agent-Driven Workflow": run_agent_workflow
    }
    
    results = {}
    for name, workflow in workflows.items():
        print(f"\nRunning {name}...")
        results[name] = workflow(blog_post)
    
    print("\nWorkflow Comparative Analysis:")
    for name, result in results.items():
        print(f"\n{name} Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    return results

def main():
    # Load sample blog post
    blog_post = get_sample_blog_post()
    
    if blog_post:
        # Run comparative workflow evaluation
        comparative_workflow_evaluation(blog_post)

if __name__ == "__main__":
    main()