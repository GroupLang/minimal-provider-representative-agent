import json
import boto3
import requests
import sys
import subprocess
import pkg_resources
from io import StringIO
from typing import List, Dict, Union, Optional, Any, Tuple, Set, Literal
import ast
import importlib.util
import os
import openai
import argparse
import pkgutil
# Standard library imports
import abc
import argparse
import ast
import asyncio
import collections
import concurrent
import contextlib
import csv
import dataclasses
import datetime
import email
import enum
import functools
import hashlib
import html
import http
import importlib.util
import io
import itertools
import json
import logging
import math
import multiprocessing
import os
import pathlib
import queue
import random
import re
import shutil
import socket
import socketserver
import sqlite3
import ssl
import string
import subprocess
import sys
import tempfile
import threading
import time
import typing
from typing import List, Dict, Union, Optional, Any, Tuple, Set, Literal
import unittest
import urllib
import xml
import zipfile

# Third-party imports
import boto3
import pkg_resources
import requests
# String IO
from io import StringIO


# Model configurations
MODEL_CONFIG = "anthropic.claude-3-5-sonnet-20241022-v2:0"
OPENAI_MODEL_CONFIG = "o3-mini"  # Default OpenAI model

# Comprehensive prompt template for code generation
CODE_GENERATION_PROMPT = """You are an expert software developer tasked with converting a Mermaid diagram into working Python code. 
Your goal is to create production-ready, well-structured code that implements the workflow described in the diagram.

Here's the Mermaid diagram to convert:

```mermaid
{mermaid_diagram}
```

Requirements:
1. Accept just a single string argument called "workflow" from the command line
2. Follow Python best practices and PEP 8 style guidelines
3. Add docstrings and comments where necessary
4. Include proper error handling and use print statements instead of logging
5. Print the workflow results at each step and the final output
6. Structure the code in a modular and maintainable way
7. Include any necessary imports and dependencies
8. Implement proper async/await patterns where appropriate
9. It can enable support for using OpenAI LLMs for code generation and processing.
10. Ensure that the code returns the final workflow output after processing.

Additional Context:
- The code should be production-ready and maintainable
- Include proper exception handling with print statements
- Implement a proper __main__ block for command-line execution
- Handle command-line arguments using argparse or sys.argv
- Provide clear usage instructions in docstrings

Available Python Packages:
- Standard library modules (os, sys, etc.)
- aiofiles (for async file operations)
- requests (for HTTP requests)
- Any additional packages needed for core functionality

Return ONLY the implementation code without any explanation or markdown formatting.
The code should be complete and ready to use."""

# Prompt template for code error fixing
CODE_FIX_PROMPT = """You are an expert software developer tasked with fixing errors in Python code.
The code was generated from a Mermaid diagram but has some execution errors.

Here's the original code:

```python
{code}
```

When executed, the code produced the following error:

```
{error}
```

Requirements:
1. Fix the code to address these errors
2. Use appropriate Python packages for the task
3. Follow Python best practices and PEP 8 style guidelines
4. Maintain existing functionality while fixing errors
5. Add proper error handling if missing

Return ONLY the fixed implementation code without any explanation or markdown formatting.
The code should be complete and ready to use."""

openai.api_key = os.getenv("OPENAI_API_KEY")

STDLIB_MODULES = {
    'abc', 'asyncio', 'dataclasses', 'datetime', 'enum', 'io', 'json', 'logging',
    'os', 'pathlib', 'sys', 'typing', 're', 'subprocess', 'importlib', 'argparse',
    'collections', 'contextlib', 'functools', 'itertools', 'math', 'random', 'string',
    'time', 'unittest', 'xml', 'zipfile', 'pathlib', 'shutil', 'tempfile', 'hashlib',
    'http', 'urllib', 'socket', 'ssl', 'email', 'sqlite3', 'csv', 'html',
    'multiprocessing', 'concurrent', 'threading', 'queue', 'socketserver'
}

def is_standard_library(module_name: str) -> bool:
    """Check if a module is part of Python's standard library."""
    if module_name in STDLIB_MODULES:
        return True
        
    try:
        # Try to find the module spec
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False
        
        # If it's in stdlib or platform specific stdlib dirs, it's standard library
        return any(
            path is not None and ('stdlib' in str(path) or 'lib-dynload' in str(path))
            for path in spec.submodule_search_locations or []
        )
    except (ImportError, AttributeError):
        return False

def extract_imports(code: str) -> Set[str]:
    """Extract import statements from Python code and return package names."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        print("Warning: Could not parse code for imports due to syntax error")
        return set()
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                base_module = name.name.split('.')[0]
                if not is_standard_library(base_module):
                    imports.add(base_module)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split('.')[0]
                if not is_standard_library(base_module):
                    imports.add(base_module)
    
    return imports

def get_installed_packages() -> Set[str]:
    """Get a set of installed Python packages."""
    return {pkg.key for pkg in pkg_resources.working_set}

async def get_credentials():
    """Fetch AWS credentials from the API endpoint."""
    print("Fetching AWS credentials...")
    try:
        response = requests.get('https://ujz6ztag95.execute-api.us-east-1.amazonaws.com/credentials')
        if not response.ok:
            print(f"Failed to fetch credentials. Status code: {response.status_code}")
            print(f"Response text: {response.text}")
            raise Exception('Failed to fetch credentials')
        credentials = response.json()
        print("Successfully fetched credentials")
        return credentials
    except Exception as error:
        print(f'Error fetching credentials: {str(error)}')
        raise error

async def invoke_model(prompt: str, provider: Literal["bedrock", "openai"] = "openai", model_id: Optional[str] = None) -> str:
    """Invoke an LLM model with a prompt and return the response.
    
    Args:
        prompt: The prompt to send to the model
        provider: The LLM provider to use ("bedrock" or "openai")
        model_id: The model ID to use (defaults to provider-specific config)
        
    Returns:
        The generated text from the model
    """
    if provider == "openai":
        return await invoke_openai_model(prompt, model_id or OPENAI_MODEL_CONFIG)
    else:
        return await invoke_bedrock_model(prompt, model_id or MODEL_CONFIG)

async def invoke_openai_model(prompt: str, model_id: str = OPENAI_MODEL_CONFIG) -> str:
    """Invoke the OpenAI model with a prompt and return the response."""
    try:
        print(f"\nInvoking OpenAI model {model_id}...")
        response = openai.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        
        if not response or not response.choices:
            raise Exception("Invalid response format from OpenAI API")
            
        generated_text = response.choices[0].message.content
        
        # Extract code block if present, otherwise return the raw text
        import re
        code_match = re.search(r'```(?:\w+)?\n([\s\S]*?)```', generated_text)
        return code_match.group(1).strip() if code_match else generated_text.strip()
        
    except Exception as error:
        print(f'Error generating text with OpenAI: {error}')
        return f"# Error: Failed to generate text\n# {str(error)}"

async def invoke_bedrock_model(prompt: str, model_id: str = MODEL_CONFIG) -> str:
    """Invoke the Bedrock model with a prompt and return the response."""
    # Get credentials and create boto3 session
    credentials_data = await get_credentials()
    session = boto3.Session(
        aws_access_key_id=credentials_data['credentials']['accessKeyId'],
        aws_secret_access_key=credentials_data['credentials']['secretAccessKey'],
        region_name=credentials_data['region']
    )
    
    # Use the region from credentials
    region = credentials_data['region']
    
    try:
        print(f"\nAttempting to use Bedrock in region: {region}")
        client = session.client('bedrock-runtime', region_name=region)
        print(f"Successfully created Bedrock client in region: {region}")
    except Exception as e:
        raise Exception(f"Failed to use region {region}: {str(e)}")

    try:
        # Prepare the payload for Claude using Messages API format
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7
        }

        print(f"\nInvoking model {model_id} in {region}...")
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload)
        )
        print("Successfully invoked the model")
        
        if 'body' not in response:
            raise Exception('Empty response received from Bedrock')
        
        response_body = response['body'].read().decode('utf-8')
        
        if not response_body:
            raise Exception('Empty response body after streaming')
        
        try:
            result = json.loads(response_body)
            
            if not result or not isinstance(result, dict):
                raise Exception('Invalid response format')

            if 'content' not in result or not result['content'] or not isinstance(result['content'], list):
                raise Exception('No content array in response')
            
            content_item = result['content'][0]
            if not isinstance(content_item, dict) or 'type' not in content_item or 'text' not in content_item:
                raise Exception('Invalid content item format')
            
            generated_text = content_item['text']

            # Extract code block if present, otherwise return the raw text
            import re
            code_match = re.search(r'```(?:\w+)?\n([\s\S]*?)```', generated_text)
            return code_match.group(1).strip() if code_match else generated_text.strip()
            
        except Exception as parse_error:
            print(f'Response parsing error: {parse_error}')
            print(f'Raw response body: {response_body}')
            return f"# Error: Failed to parse response\n# {str(parse_error)}"
            
    except Exception as error:
        print(f'Error generating text: {error}')
        return f"# Error: Failed to generate text\n# {str(error)}"

async def get_real_package_name(module_name: str) -> str:
    """Query LLM to get the real package name for a Python module.
    
    Args:
        module_name: The module name as used in import statements
        
    Returns:
        The real package name to use with pip
    """
    prompt = f"""
    What is the correct pip package name to install for the Python module '{module_name}'?
    Return ONLY the package name, nothing else. If the module name is the same as the package name,
    just return that name. If it's a standard library module that doesn't need installation, return 'stdlib'.
    """
    
    try:
        # Use OpenAI for simple queries
        response = await invoke_openai_model(prompt)
        # Clean up the response
        package_name = response.strip().lower()
        
        # If the LLM indicates it's a standard library module
        if package_name == 'stdlib':
            return None
            
        return package_name
    except Exception as e:
        print(f"Error getting real package name for {module_name}: {str(e)}")
        # Fall back to the original module name
        return module_name

async def install_dependencies(code: str) -> Tuple[bool, str]:
    """Install required Python packages for the generated code.
    
    Args:
        code: The Python code to analyze for dependencies
        
    Returns:
        Tuple containing:
        - Success flag (True if all installations were successful)
        - Output message or error details
    """
    try:
        # Extract required packages from imports
        required_modules = extract_imports(code)
        if not required_modules:
            return True, "No third-party packages required."
        
        # Get currently installed packages
        installed_packages = get_installed_packages()
        
        # Get real package names for each module
        packages_to_install = []
        for module_name in required_modules:
            # Skip if already installed
            if module_name in installed_packages:
                continue
                
            # Get the real package name
            real_package_name = await get_real_package_name(module_name)
            if real_package_name and real_package_name not in installed_packages:
                packages_to_install.append(real_package_name)
        
        if not packages_to_install:
            return True, "All required packages are already installed."
        
        # Install missing packages
        print(f"Installing required packages: {', '.join(packages_to_install)}")
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", *packages_to_install],
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            return True, f"Successfully installed packages: {', '.join(packages_to_install)}"
        else:
            # If installation fails, try installing packages one by one
            print("Batch installation failed. Trying to install packages one by one...")
            failed_packages = []
            
            for package in packages_to_install:
                print(f"Installing {package}...")
                single_process = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True
                )
                
                if single_process.returncode != 0:
                    failed_packages.append(package)
                    print(f"Failed to install {package}: {single_process.stderr}")
            
            if failed_packages:
                return False, f"Failed to install some packages: {', '.join(failed_packages)}"
            else:
                return True, "Successfully installed all packages individually."
            
    except Exception as e:
        return False, f"Error during dependency installation: {str(e)}"

async def execute_generated_code(code: str) -> Tuple[bool, str, str]:
    """Execute the generated Python code in a controlled environment using a temporary file.
    
    Args:
        code: The Python code to execute as a string
        
    Returns:
        Tuple containing:
        - Success flag (True if execution was successful, False otherwise)
        - Standard output captured during execution
        - Error output or exception message if execution failed
    """
    # Create a temporary file to store the generated code
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
        temp_file_path = temp_file.name
        
        # Add argument handling to the generated code
        argument_handling = """
import argparse

def parse_arguments():
    \"\"\"Parse command line arguments.\"\"\"
    parser = argparse.ArgumentParser(description="Process workflow")
    parser.add_argument("workflow", type=str, help="Workflow string to process")
    return parser.parse_args()

# Get command line arguments
args = parse_arguments()
workflow = args.workflow
"""
        
        # Write the code to the temporary file
        temp_file.write(argument_handling + "\n\n" + code)
    
    success = True
    stdout_output = ""
    stderr_output = ""
    
    try:
        print(f"Executing generated code from temporary file: {temp_file_path}\n")
        
        # Get the current command line arguments to pass to the subprocess
        workflow_arg = sys.argv[1] if len(sys.argv) > 1 else "default workflow"
        
        # Execute the temporary file as a subprocess
        process = subprocess.run(
            [sys.executable, temp_file_path, workflow_arg],
            capture_output=True,
            text=True
        )
        
        # Capture stdout and stderr
        stdout_output = process.stdout
        stderr_output = process.stderr
        
        # Check if the process was successful
        if process.returncode != 0:
            success = False
            print(f"Process exited with code {process.returncode}")
        else:
            print("\nCode execution completed successfully.")
            
    except Exception as e:
        success = False
        stderr_output = f"Error during code execution: {str(e)}"
        print(stderr_output)
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
            print(f"Temporary file {temp_file_path} removed")
        except Exception as e:
            print(f"Warning: Failed to remove temporary file {temp_file_path}: {str(e)}")
    
    return success, stdout_output, stderr_output

async def split_workflow_into_subtasks(workflow_message: str, execution_logs: str) -> List[Dict[str, Any]]:
    """
    Send the workflow message and execution logs to ChatGPT to split the workflow into subtasks.
    
    Args:
        workflow_message: The original workflow message
        execution_logs: The logs captured during code execution
        
    Returns:
        A list of dictionaries, each representing a subtask with properties like
        name, description, dependencies, complexity, status, and estimated time
    """
    prompt = f"""
    I have a workflow described as: "{workflow_message}"
    
    Here are the execution logs from running this workflow:
    
    {execution_logs}
    
    Based on the initial workflow and these execution logs, please split this workflow into small, 
    manageable subtasks. For each subtask, follow this EXACT format:

    ```
    ## Subtask [number]: [Name]
    - **Description**: [Brief description of what needs to be done]
    - **Dependencies**: [List of dependent subtasks by number, or "None" if no dependencies]
    - **Complexity**: [Low/Medium/High]
    - **Status**: Not Started
    - **Estimated Time**: [Time estimate in hours]
    ```

    Ensure each subtask is:
    1. Clearly defined with a specific outcome
    2. Small enough to be completed in a reasonable timeframe
    3. Properly sequenced with accurate dependencies
    4. Assigned an appropriate complexity level
    
    Start with an overview section that summarizes the total number of subtasks and estimated completion time.
    """
    
    try:
        # Get the markdown response from the LLM
        markdown_response = await invoke_openai_model(prompt)
        
        # Parse the markdown to extract subtasks
        subtasks = []
        
        # Use regex to find subtask sections
        subtask_pattern = r'## Subtask (\d+): (.*?)\n(.*?)(?=\n## Subtask|\Z)'
        subtask_matches = re.finditer(subtask_pattern, markdown_response, re.DOTALL)
        
        for match in subtask_matches:
            number = int(match.group(1))
            name = match.group(2).strip()
            content = match.group(3).strip()
            
            # Extract properties using regex
            description_match = re.search(r'\*\*Description\*\*: (.*?)(?=\n-|\Z)', content)
            dependencies_match = re.search(r'\*\*Dependencies\*\*: (.*?)(?=\n-|\Z)', content)
            complexity_match = re.search(r'\*\*Complexity\*\*: (.*?)(?=\n-|\Z)', content)
            status_match = re.search(r'\*\*Status\*\*: (.*?)(?=\n-|\Z)', content)
            time_match = re.search(r'\*\*Estimated Time\*\*: (.*?)(?=\n-|\Z)', content)
            
            # Create subtask dictionary
            subtask = {
                "number": number,
                "name": name,
                "description": description_match.group(1).strip() if description_match else "",
                "dependencies": dependencies_match.group(1).strip() if dependencies_match else "None",
                "complexity": complexity_match.group(1).strip() if complexity_match else "Medium",
                "status": status_match.group(1).strip() if status_match else "Not Started",
                "estimated_time": time_match.group(1).strip() if time_match else "1 hour"
            }
            
            subtasks.append(subtask)
        
        # If no subtasks were found, return an empty list with a warning
        if not subtasks:
            print("Warning: No subtasks could be parsed from the LLM response")
            
        return subtasks
        
    except Exception as e:
        print(f"Error splitting workflow into subtasks: {str(e)}")
        return []  # Return empty list on error

async def generate_code_from_mermaid(mermaid_diagram: str, provider: Literal["bedrock", "openai"] = "openai") -> str:
    """Generate Python code that implements a workflow described in a Mermaid diagram.
    
    Args:
        mermaid_diagram: The Mermaid diagram syntax describing the workflow
        provider: The LLM provider to use ("bedrock" or "openai")
        
    Returns:
        Generated Python code implementing the workflow
    """
    prompt = CODE_GENERATION_PROMPT.format(mermaid_diagram=mermaid_diagram)
    return await invoke_model(prompt, provider)

async def fix_code_errors(code: str, error_output: str, provider: Literal["bedrock", "openai"] = "openai") -> str:
    """Fix errors in generated code by submitting it back to the LLM.
    
    Args:
        code: The original code with errors
        error_output: The error output from executing the code
        provider: The LLM provider to use ("bedrock" or "openai")
        
    Returns:
        Fixed code from the LLM
    """
    print("\n" + "="*80)
    print("FIXING CODE ERRORS")
    print("="*80)
    
    prompt = CODE_FIX_PROMPT.format(code=code, error=error_output)
    return await invoke_model(prompt, provider)

async def process_message(message: str, chat_history: str = None) -> str:    
    # Check if workflow_code.py exists, if not generate it from a default mermaid diagram
    if not os.path.exists("workflow_code.py"):
        print("workflow_code.py not found. Please create it first.")
        return "Error: workflow_code.py not found. Please create it first."

    # Load generated code from code.py file
    with open("workflow_code.py", "r") as file:
        generated_code = file.read()
    
    install_success, install_message = await install_dependencies(generated_code)
    print(install_message)
    
    if not install_success:
        print("Failed to install required dependencies. Skipping code execution.")
        return "Failed to install required dependencies."
    
    # Execute the generated code
    print("\n" + "="*80)
    print("EXECUTING GENERATED CODE")
    print("="*80)
    
    # Pass a workflow argument for creating a frontend app
    workflow_arg = message
    sys.argv = ['script.py', workflow_arg]  # Simulate command line argument
    success, stdout, stderr = await execute_generated_code(generated_code)
    
    print("\n" + "="*80)
    print("EXECUTION RESULTS")
    print("="*80)
    
    if success:
        print("Code executed successfully!")
    else:
        print("Code execution failed!")
        
        # Initialize variables for iterative fixing
        current_code = generated_code
        max_iterations = 10  # Increased from 3 to 10 for more fix attempts
        iteration = 0
        
        # Try fixing the code up to max_iterations times
        while not success and iteration < max_iterations:
            iteration += 1
            print(f"\n" + "="*80)
            print(f"ATTEMPTING FIX ITERATION {iteration}/{max_iterations}")
            print("="*80)
            
            # Try to fix the code
            current_code = await fix_code_errors(current_code, stderr or stdout)
            
            print("\nFIXED CODE:")
            print("-"*40)
            print(current_code)
            
            # Save the fixed code
            with open("workflow_code.py", "w") as file:
                file.write(current_code)
            
            # Install dependencies for fixed code
            print("\n" + "="*80)
            print(f"CHECKING AND INSTALLING DEPENDENCIES FOR ITERATION {iteration}")
            print("="*80)
            
            install_success, install_message = await install_dependencies(current_code)
            print(install_message)
            
            if not install_success:
                print(f"Failed to install required dependencies in iteration {iteration}. Skipping execution.")
                break
            
            # Execute the fixed code
            print("\n" + "="*80)
            print(f"EXECUTING FIXED CODE (ITERATION {iteration})")
            print("="*80)
            
            success, stdout, stderr = await execute_generated_code(current_code)
            
            print("\n" + "="*80)
            print(f"ITERATION {iteration} EXECUTION RESULTS")
            print("="*80)
            
            if success:
                print(f"Code fixed successfully after {iteration} iteration(s)!")
            else:
                print(f"Code execution failed in iteration {iteration}")
        
        if not success:
            print(f"\nFailed to fix code after {max_iterations} iterations")
        
    print("\nSTANDARD OUTPUT:")
    print("-"*40)
    print(stdout)
    
    if stderr:
        print("\nERROR OUTPUT:")
        print("-"*40)
        print(stderr)
    
    # Split the workflow into subtasks using ChatGPT
    print("\n" + "="*80)
    print("SPLITTING WORKFLOW INTO SUBTASKS")
    print("="*80)
    
    # Combine stdout and stderr for complete logs
    execution_logs = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    subtasks = await split_workflow_into_subtasks(message, execution_logs)
    
    print("\nWORKFLOW SUBTASKS:")
    print("-"*40)
    
    # Convert subtasks list to formatted markdown for display and saving
    subtasks_markdown = f"# Workflow Subtasks for: {message}\n\n"
    
    if subtasks:
        # Add overview section
        subtasks_markdown += "## Overview\n"
        subtasks_markdown += f"- Total subtasks: {len(subtasks)}\n"
        total_hours = sum(float(task['estimated_time'].replace(' hours', '').replace(' hour', '')) 
                          for task in subtasks 
                          if task['estimated_time'].replace(' hours', '').replace(' hour', '').replace('.', '', 1).isdigit())
        subtasks_markdown += f"- Estimated completion time: {total_hours:.1f} hours\n\n"
        
        # Add each subtask
        for task in subtasks:
            subtasks_markdown += f"## Subtask {task['number']}: {task['name']}\n"
            subtasks_markdown += f"- **Description**: {task['description']}\n"
            subtasks_markdown += f"- **Dependencies**: {task['dependencies']}\n"
            subtasks_markdown += f"- **Complexity**: {task['complexity']}\n"
            subtasks_markdown += f"- **Status**: {task['status']}\n"
            subtasks_markdown += f"- **Estimated Time**: {task['estimated_time']}\n\n"
    else:
        subtasks_markdown += "No subtasks were identified.\n"
    
    print(subtasks_markdown)
    
    # Save subtasks to a file
    with open("workflow_subtasks.md", "w") as file:
        file.write(subtasks_markdown)
    
    print("\nSubtasks saved to workflow_subtasks.md")
    
    return f"Workflow processed. {len(subtasks)} subtasks identified and saved to workflow_subtasks.md"

async def main():
    # Example Mermaid diagram representing a project development workflow
    if len(sys.argv) > 1:
        # If arguments are provided, process the workflow message
        workflow_message = sys.argv[1]
        result = await process_message(workflow_message)
        print(result)
    else:
        # Example usage with a default workflow
        parser = argparse.ArgumentParser(description="Process a workflow message")
        parser.add_argument("workflow", nargs="?", default="Create a modern React frontend application with responsive design, routing, and state management", 
                          help="Workflow message to process")
        args = parser.parse_args()
        
        result = await process_message(args.workflow)
        print(result)

if __name__ == "__main__":
    # run the asyncio event loop
    import asyncio
    asyncio.run(main())
