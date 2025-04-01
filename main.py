from mcp.server.fastmcp import FastMCP, Context
from typing import Dict, List, Optional, Any
import asyncio
from playwright.async_api import async_playwright
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import re
from bs4 import BeautifulSoup

# Create an MCP server
mcp = FastMCP("BrowserFormFiller")
mcp = FastMCP("BrowserFormFiller", dependencies=["playwright", "langchain", "langchain_openai", "beautifulsoup4"])
USER_AGENT="form-automate-app/1.0"

# Initialize the LLM
def get_llm():
    # Get API key from environment variable - this should be set before running the server
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key
    )

# Define the prompt for extracting form field information
form_analysis_prompt = ChatPromptTemplate.from_template("""
    You are an expert in HTML form analysis. Extract all form fields from the provided HTML and also the submit button and return them strictly as a JSON array.

    HTML Content:
    {html_content}

    Respond with only a valid JSON array that contains a name, type, label and xpath for each form field, no explanations, no formatting, and no Markdown.
    for the submit button, It would also have name, type, label and xpath. The type and label will be 'submit', while the name will be the "text" in the submit tag, xpath will be the xpath to click the button.
    """
)

# Define the prompt for generating a form submission strategy
form_filling_prompt = ChatPromptTemplate.from_template("""
    You are an expert in browser automation. You have the following form fields on a webpage:

    {form_fields}

    And you have the following data to fill in the form:

    {form_data}

    For each field that matches the provided data, provide the XPath and the value to fill.
    If there's a field in the form data that doesn't match any form field name exactly,
    suggest the best match based on the label or name.

    Respond with only a valid JSON array that contains the xpath, value and action(either fill, select or check).


    For the submit button, include an entry with action "click" and the XPath of the submit button.
    For radio buttons and checkboxes, use true/false values.
    For select dropdowns, provide the visible text of the option to select.
"""
)

@mcp.tool()
async def fill_form(url: str, form_data: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
    """
    Fill a form on a webpage and submit it. It can be a login form, subscription form, etc.
    
    Args:
        url: The URL of the webpage with the form
        form_data: Dictionary of form field names and their values to fill
    """

    ctx.info(f"Starting form automation for: {url}")
    ctx.info(f"Form data to fill: {form_data}")
    
    results = {
        "success": False,
        "url": url,
        "fields_filled": [],
        "errors": [],
        "screenshot": None
    }
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, timeout=120000)
            ctx.info("Browser launched")
            print("Browser launched")
            
            page = await browser.new_page()
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            ctx.info("Page loaded")
            
            # Get the HTML content for analysis
            html_content = await page.content()
            ctx.info(f"html_content : {html_content}")
          
            # Use LLM to analyze the form
            llm = get_llm()
            ctx.info("Analyzing form structure with LLM")

            form_analysis_response = llm.invoke(form_analysis_prompt.format(html_content=html_content))
            ctx.info(f"Raw filling strategy response: {form_analysis_response}")
            form_fields = form_analysis_response.content

            ctx.info(f"Raw form analysis response: {form_fields}")
            

            # Extract JSON from the response if needed
            json_match = re.search(r'```json\n(.*?)```', form_fields, re.DOTALL)
            if json_match:
                ctx.info(f"Processed response: {json_match}")
                form_fields = json_match.group(1)
            
            ctx.info("Form fields identified")
            
            # Generate filling strategy
            ctx.info("Generating form filling strategy")
            filling_strategy_response = await llm.ainvoke(form_filling_prompt.format(
                form_fields=form_fields,
                form_data=str(form_data)
            ))
            
            filling_strategy = filling_strategy_response.content
            print("filling_strategy", "\n", filling_strategy)

            # Extract JSON from the response if needed
            json_match = re.search(r'```json\n(.*?)```', filling_strategy, re.DOTALL)
            if json_match:
                filling_strategy = json_match.group(1)
            
            # We need to convert the filling_strategy from string to a list of dictionaries
            # In a production environment, you would use proper JSON parsing
            import json
            try:
                filling_actions = json.loads(filling_strategy)
            except json.JSONDecodeError:
                ctx.info("Error parsing filling strategy, attempting alternative parsing")
                # Fallback to safe eval
                import ast
                filling_actions = ast.literal_eval(filling_strategy)
            
            # Execute the filling strategy
            ctx.info(f"Executing filling strategy with {len(filling_actions)} actions")
            for i, action in enumerate(filling_actions):
                try:
                    xpath = action.get("xpath")
                    value = action.get("value")
                    action_type = action.get("action", "fill")
                    
                    if xpath and action_type == "fill" and value is not None:
                        await page.fill(xpath, str(value))
                        results["fields_filled"].append(f"Filled {xpath} with '{value}'")
                        ctx.info(f"Filled {xpath}")
                        
                    elif xpath and action_type == "select" and value is not None:
                        await page.select_option(xpath, label=str(value))
                        results["fields_filled"].append(f"Selected '{value}' in {xpath}")
                        ctx.info(f"Selected option in {xpath}")
                        
                    elif xpath and action_type == "check":
                        if value:
                            await page.check(xpath)
                            results["fields_filled"].append(f"Checked {xpath}")
                            ctx.info(f"Checked {xpath}")
                        else:
                            await page.uncheck(xpath)
                            results["fields_filled"].append(f"Unchecked {xpath}")
                            ctx.info(f"Unchecked {xpath}")
                            
                    elif xpath and action_type == "click":
                        # This is likely the submit button
                        await page.wait_for_timeout(1000)  # Small delay before submission
                        
                        # Take screenshot before submitting
                        screenshot_path = f"/tmp/form_screenshot_{hash(url) % 1000}.png"
                        await page.screenshot(path=screenshot_path)
                        ctx.info(f"Screenshot taken before submission: {screenshot_path}")
                        
                        # Click the submit button
                        await page.click(xpath)
                        ctx.info(f"Clicked {xpath} (submit button)")
                        
                        # Wait for navigation or network idle
                        try:
                            await page.wait_for_load_state("networkidle", timeout=5000)
                            ctx.info("Page navigation completed after form submission")
                        except Exception as e:
                            ctx.info(f"Navigation timeout after submission: {str(e)}")
                            
                        results["fields_filled"].append(f"Clicked submit button {xpath}")
                except Exception as e:
                    error_msg = f"Error with action {i+1} ({action_type} on {xpath}): {str(e)}"
                    results["errors"].append(error_msg)
                    ctx.info(error_msg)
            
            # Final screenshot after form submission
            final_screenshot_path = f"/tmp/form_result_{hash(url) % 1000}.png"
            await page.screenshot(path=final_screenshot_path)
            ctx.info(f"Final screenshot taken: {final_screenshot_path}")
            
            # Get current URL after submission
            current_url = page.url
            results["final_url"] = current_url
            
            # Capture page content after submission
            final_content = await page.content()
            soup = BeautifulSoup(final_content, 'html.parser')
            
            # Look for confirmation messages or errors
            confirmation_texts = []
            for tag in soup.find_all(['div', 'p', 'h1', 'h2', 'h3', 'h4', 'span']):
                text = tag.get_text(strip=True)
                classes = tag.get('class', [])
                if text and any(keyword in text.lower() for keyword in ['success', 'thank', 'confirm', 'receipt']):
                    confirmation_texts.append(text)
                elif classes and any(cls for cls in classes if any(keyword in cls.lower() for keyword in ['success', 'confirm', 'thank'])):
                    confirmation_texts.append(text)
            
            if confirmation_texts:
                results["confirmation_messages"] = confirmation_texts
                ctx.info(f"Found confirmation messages: {confirmation_texts}")
            
            # input("Enter to continue ...")
            await browser.close()
            
            # Set success based on whether we had errors
            results["success"] = len(results["errors"]) == 0
            
            
            return results
            
    except Exception as e:
        results["errors"].append(f"General error: {str(e)}")
        ctx.info(f"Error during form automation: {str(e)}")
        return results
    

# If running directly, start the server
if __name__ == "__main__":
    mcp.run()