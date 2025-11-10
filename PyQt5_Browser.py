from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter, QTextEdit, QPushButton, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QLabel, QListWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl, QTimer, QDateTime
import sys, requests, json
from bs4 import BeautifulSoup
import markdown
import re

# Import LLM configuration
from llm_config import get_api_endpoint, get_api_key, get_model, get_headers

class AIBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My AI Browser")
        self.resize(1400, 900)
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize request tracking
        self.current_request = None
        self.expected_info_type = None
        self.request_fulfilled = False
        self.navigation_attempts = 0
        self.max_navigation_attempts = 3
        
        # Create address bar
        self.address_bar = QLineEdit()
        self.address_bar.returnPressed.connect(self.load_url)
        self.address_bar.setText("https://www.skynews.com")
        
        # Create navigation button
        self.go_btn = QPushButton("Go")
        self.go_btn.clicked.connect(self.load_url)
        
        # Create web view
        self.web = QWebEngineView()
        self.web.load(QUrl("https://www.skynews.com"))
        self.web.urlChanged.connect(self.update_address_bar)
        
        # Create AI command interface
        self.ai_command = QTextEdit()
        self.ai_command.setPlaceholderText("Enter AI commands (e.g., 'check the iPhone 17 max price')...")
        self.ai_command.setMaximumHeight(60)
        self.ai_execute_btn = QPushButton("Execute AI Command")
        self.ai_execute_btn.clicked.connect(self.execute_ai_command)
        
        # Create AI actions log
        self.actions_log = QListWidget()
        self.actions_log.setMaximumHeight(800)
        
        # Create chat interface
        self.chat = QTextEdit()
        self.chat.setPlaceholderText("Ask your AI...")
        self.chat.setAcceptRichText(True)
        self.send_btn = QPushButton("Ask AI")
        self.send_btn.clicked.connect(self.ask_ai)
        
        # Layout for address bar
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.address_bar)
        nav_layout.addWidget(self.go_btn)
        
        # Layout for chat
        chat_layout = QVBoxLayout()
        chat_layout.addWidget(QLabel("AI Command:"))
        chat_layout.addWidget(self.ai_command)
        chat_layout.addWidget(self.ai_execute_btn)
        chat_layout.addWidget(QLabel("AI Actions Log:"))
        chat_layout.addWidget(self.actions_log)
        chat_widget = QWidget()
        chat_widget.setLayout(chat_layout)
        
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.web)
        
        splitter = QSplitter(Qt.Horizontal)
        web_widget = QWidget()
        web_widget.setLayout(main_layout)
        splitter.addWidget(web_widget)
        splitter.addWidget(chat_widget)
        splitter.setSizes([1000, 400])
        
        main_widget.setLayout(QVBoxLayout())
        main_widget.layout().addWidget(splitter)
        self.setCentralWidget(main_widget)
    
    
    def load_url(self):
        url = self.address_bar.text()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        self.web.load(QUrl(url))
    
    def update_address_bar(self, url):
        self.address_bar.setText(url.toString())
    
    def add_action_log(self, action):
        """Add an action to the actions log"""
        self.actions_log.addItem(action)
        self.actions_log.scrollToBottom()
        
        # Add to conversation history
        self.conversation_history.append({
            "type": "action",
            "content": action,
            "timestamp": QDateTime.currentDateTime().toString()
        })
    
    def add_conversation_message(self, role, message):
        """Add a message to the conversation history"""
        self.conversation_history.append({
            "type": "conversation",
            "role": role,  # "user" or "ai"
            "content": message,
            "timestamp": QDateTime.currentDateTime().toString()
        })
    
    def generate_conversational_response_with_llm(self, user_message):
        """Use LLM to generate a conversational response"""
        self.add_action_log(f"Generating conversational response with LLM")
        
        # Build conversation history context
        conversation_context = ""
        for msg in self.conversation_history[-10:]:  # Last 10 messages
            if msg["type"] == "conversation":
                role = "User" if msg["role"] == "user" else "AI"
                conversation_context += f"{role}: {msg['content']}\n"
        
        # Create a prompt for the LLM to generate a conversational response
        prompt = f"""
You are an AI assistant in a smart browser. Generate a natural, conversational response to the user's message.

Conversation history:
{conversation_context}

User's latest message: "{user_message}"

Please provide a helpful, conversational response. Be friendly and informative.
"""
        
        # Call the LLM API using centralized configuration
        api_endpoint = get_api_endpoint()
        headers = get_headers()
        
        payload = {
            "model": get_model(),
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the response from the LLM
            llm_response = result["choices"][0]["text"].strip()
            
            self.add_conversation_message("ai", llm_response)
            return llm_response
            
        except Exception as e:
            self.add_action_log(f"Error in LLM conversational response: {str(e)}")
            fallback_response = "I'm here to help! How can I assist you with your browsing today?"
            self.add_conversation_message("ai", fallback_response)
            return fallback_response
    
    def handle_error_with_llm(self, error_message, context):
        """Use LLM to analyze errors and suggest solutions"""
        self.add_action_log(f"Handling error with LLM: {error_message}")
        
        # Create a prompt for the LLM to analyze the error
        prompt = f"""
Analyze the following error and suggest a solution:

Error: "{error_message}"

Context:
{context}

Please provide your response in the following JSON format:
{{
  "analysis": "Brief analysis of what went wrong",
  "suggested_solution": "Suggested solution to fix the error",
  "confidence": "How confident you are in your solution (0.0 to 1.0)"
}}

Just provide the JSON response, nothing else.
"""
        
        # Call the LLM API using centralized configuration
        api_endpoint = get_api_endpoint()
        headers = get_headers()
        
        payload = {
            "model": get_model(),
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.3,
            "stream": False
        }
        
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the JSON response from the LLM
            llm_response = result["choices"][0]["text"].strip()
            
            # Parse the JSON response
            import json
            parsed_response = json.loads(llm_response)
            
            analysis = parsed_response["analysis"]
            suggested_solution = parsed_response["suggested_solution"]
            confidence = parsed_response["confidence"]
            
            self.add_action_log(f"LLM Error Analysis - Analysis: {analysis}")
            self.add_action_log(f"Suggested Solution: {suggested_solution} (Confidence: {confidence})")
            
            return {
                "analysis": analysis,
                "suggested_solution": suggested_solution,
                "confidence": confidence
            }
            
        except Exception as e:
            self.add_action_log(f"Error in LLM error analysis: {str(e)}")
            return None
    
    def generate_progress_update_with_llm(self, current_step, total_steps, task_description):
        """Use LLM to generate natural language progress updates"""
        self.add_action_log(f"Generating progress update with LLM: {current_step}/{total_steps}")
        
        # Create a prompt for the LLM to generate a progress update
        prompt = f"""
Generate a natural language progress update for a task:

Task: "{task_description}"
Current step: {current_step}
Total steps: {total_steps}

Please provide your response in the following JSON format:
{{
  "progress_message": "A natural language progress update message",
  "percentage": "The completion percentage (0-100)"
}}

Just provide the JSON response, nothing else.
"""
        
        # Call the LLM API using centralized configuration
        api_endpoint = get_api_endpoint()
        headers = get_headers()
        
        payload = {
            "model": get_model(),
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.3,
            "stream": False
        }
        
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the JSON response from the LLM
            llm_response = result["choices"][0]["text"].strip()
            
            # Parse the JSON response
            import json
            parsed_response = json.loads(llm_response)
            
            progress_message = parsed_response["progress_message"]
            percentage = parsed_response["percentage"]
            
            self.add_action_log(f"LLM Progress Update: {progress_message} ({percentage}%)")
            
            return {
                "message": progress_message,
                "percentage": percentage
            }
            
        except Exception as e:
            self.add_action_log(f"Error in LLM progress update: {str(e)}")
            # Fallback to a simple progress message
            percentage = int((current_step / total_steps) * 100) if total_steps > 0 else 0
            return {
                "message": f"Completed step {current_step} of {total_steps}",
                "percentage": percentage
            }
    
    def navigate_to_url(self, url):
        """Navigate to a specific URL"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        self.add_action_log(f"Navigating to: {url}")
        self.web.load(QUrl(url))
        # Reset navigation attempts when navigating to a new URL
        self.navigation_attempts = 0
        # After navigation, analyze the page content
        QTimer.singleShot(3000, self.analyze_current_page)
    
    def navigate_to_url_and_wait(self, url, callback=None):
        """Navigate to a specific URL and optionally call a callback after navigation"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        self.add_action_log(f"Navigating to: {url}")
        self.web.load(QUrl(url))
        # Reset navigation attempts when navigating to a new URL
        self.navigation_attempts = 0
        
        # After navigation, analyze the page content
        def after_navigation():
            self.analyze_current_page()
            if callback:
                callback()
                
        QTimer.singleShot(3000, after_navigation)
    
    def analyze_current_page(self):
        """Analyze the current page to determine if it contains the requested information"""
        # Only analyze if we have a current request and it hasn't been fulfilled
        if not self.current_request or self.request_fulfilled:
            return
            
        self.add_action_log(f"Analyzing current page for request: {self.current_request}")
        
        def process_content(html):
            # Parse HTML and check if the page contains the requested information
            soup = BeautifulSoup(html, 'html.parser')
            text_content = soup.get_text().lower()
            
            # Check if the current page contains the requested item
            item = self.current_request.get("item", "")
            info_type = self.current_request.get("info_type", "general")
            
            # Use LLM to determine if the page contains the requested information
            prompt = f"""
Analyze the following web page content to determine if it contains information about "{item}" of type "{info_type}".

Web page content:
{text_content[:2000]}  # Limit content to 2000 characters

Please provide your response in the following JSON format:
{{
  "contains_requested_info": "Whether the page contains the requested information (true/false)",
  "confidence": "How confident you are in your assessment (0.0 to 1.0)",
  "explanation": "Brief explanation of your reasoning"
}}

Just provide the JSON response, nothing else.
"""
            
            # Call the LLM API using centralized configuration
            api_endpoint = get_api_endpoint()
            headers = get_headers()
            
            payload = {
                "model": get_model(),
                "prompt": prompt,
                "max_tokens": 300,
                "temperature": 0.3,
                "stream": False
            }
            
            try:
                response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # Extract the JSON response from the LLM
                llm_response = result["choices"][0]["text"].strip()
                
                # Parse the JSON response
                import json
                parsed_response = json.loads(llm_response)
                
                contains_info = parsed_response["contains_requested_info"]
                confidence = parsed_response["confidence"]
                explanation = parsed_response["explanation"]
                
                self.add_action_log(f"Page Analysis - Contains Info: {contains_info}, Confidence: {confidence}")
                self.add_action_log(f"Explanation: {explanation}")
                
                # If the page contains the requested information, extract it
                if contains_info and confidence > 0.7:
                    self.add_action_log(f"Page contains requested information about {item}")
                    self.extract_information(info_type)
                    self.request_fulfilled = True
                else:
                    # If the page doesn't contain the requested information, generate a new plan
                    self.add_action_log(f"Page does not contain sufficient information about {item}")
                    self.navigation_attempts += 1
                    
                    # Check if we've exceeded the maximum navigation attempts
                    if self.navigation_attempts >= self.max_navigation_attempts:
                        self.add_action_log(f"Maximum navigation attempts ({self.max_navigation_attempts}) reached. Giving up.")
                        self.add_action_log(f"Could not find information about {item} after {self.navigation_attempts} attempts.")
                        return
                    
                    # Generate a new plan to find the information
                    self.generate_alternative_plan(item, info_type)
                    
            except Exception as e:
                self.add_action_log(f"Error in page analysis: {str(e)}")
                # Fall back to a simpler approach
                self.fallback_page_analysis(item, info_type, text_content)
        
        self.get_page_content(process_content)
    
    def fallback_page_analysis(self, item, info_type, text_content):
        """Fallback method for page analysis when LLM fails"""
        self.add_action_log(f"Fallback page analysis for: {item} ({info_type})")
        
        # Simple keyword-based analysis
        item_lower = item.lower()
        text_lower = text_content.lower()
        
        # Check if the item is mentioned in the text
        if item_lower in text_lower:
            self.add_action_log(f"Found mention of '{item}' in page content")
            # Check if the info type is also mentioned
            if info_type in text_lower or info_type == "general":
                self.add_action_log(f"Page likely contains {info_type} information for '{item}'")
                self.extract_information(info_type)
                self.request_fulfilled = True
                return
        
        self.add_action_log(f"Page does not contain sufficient information about '{item}'")
        self.navigation_attempts += 1
        
        # Check if we've exceeded the maximum navigation attempts
        if self.navigation_attempts >= self.max_navigation_attempts:
            self.add_action_log(f"Maximum navigation attempts ({self.max_navigation_attempts}) reached. Giving up.")
            self.add_action_log(f"Could not find information about {item} after {self.navigation_attempts} attempts.")
            return
        
        # Generate a new plan to find the information
        self.generate_alternative_plan(item, info_type)
    
    def generate_alternative_plan(self, item, info_type):
        """Generate an alternative plan when the current page doesn't contain the requested information"""
        self.add_action_log(f"Generating alternative plan for: {item} ({info_type})")
        
        # Use LLM to create a new plan based on the current page content
        def process_content(html):
            # Extract text content for LLM analysis
            soup = BeautifulSoup(html, 'html.parser')
            text_content = soup.get_text()[:2000]  # Limit content to 2000 characters
            
            # Use LLM to create a new execution plan
            prompt = f"""
Analyze the current web page content and create an alternative execution plan to find information about "{item}" of type "{info_type}".
The current page does not contain the requested information, so we need to find another approach.

Current page content:
{text_content}

Current page URL: {self.web.url().toString()}

Please provide your response in the following JSON format:
{{
  "strategy": "The overall strategy to use (e.g., 'search_engine', 'click_link', 'form_submission', 'multi_step_process')",
  "steps": [
    {{
      "action": "The action to take (e.g., 'navigate', 'search', 'fill_form', 'click', 'extract')",
      "description": "A detailed description of what to do",
      "parameters": {{
        "url": "URL to navigate to (if applicable)",
        "query": "Search query (if applicable)",
        "form_data": {{}},  # Form data to fill (if applicable)
        "element_description": "Description of element to click (if applicable)",
        "keywords": "Keywords to search for (if applicable)"
      }}
    }}
  ],
  "confidence": "How confident you are in this plan (0.0 to 1.0)"
}}

Examples:
For finding the price of an iPhone when on an unrelated page:
{{
  "strategy": "search_engine",
  "steps": [
    {{
      "action": "navigate",
      "description": "Go to Google search",
      "parameters": {{
        "url": "https://www.google.com"
      }}
    }},
    {{
      "action": "fill_form",
      "description": "Fill search form with iPhone query",
      "parameters": {{
        "form_data": {{"q": "iPhone 17 price"}}
      }}
    }},
    {{
      "action": "click",
      "description": "Click search button",
      "parameters": {{
        "element_description": "Google Search button"
      }}
    }}
  ],
  "confidence": 0.9
}}

Just provide the JSON response, nothing else.
"""
            
            # Call the LLM API using centralized configuration
            api_endpoint = get_api_endpoint()
            headers = get_headers()
            
            payload = {
                "model": get_model(),
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.3,
                "stream": False
            }
            
            try:
                response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # Extract the JSON response from the LLM
                llm_response = result["choices"][0]["text"].strip()
                
                # Parse the JSON response
                import json
                parsed_response = json.loads(llm_response)
                
                strategy = parsed_response["strategy"]
                steps = parsed_response["steps"]
                confidence = parsed_response["confidence"]
                
                self.add_action_log(f"LLM Alternative Plan - Strategy: {strategy}, Confidence: {confidence}")
                
                # Create a plan based on the LLM response
                plan = {
                    "strategy": strategy,
                    "steps": [],
                    "confidence": confidence
                }
                
                # Process each step
                for i, step in enumerate(steps):
                    action = step["action"]
                    description = step["description"]
                    parameters = step["parameters"]
                    
                    # Add step to plan
                    plan["steps"].append({
                        "action": action,
                        "description": description,
                        "parameters": parameters
                    })
                    
                    self.add_action_log(f"  Step {i+1}: {action} - {description}")
                
                # Execute the new plan
                self.execute_plan_step(plan)
                
            except Exception as e:
                self.add_action_log(f"Error in LLM alternative planning: {str(e)}")
                # Fall back to a simple search approach
                self.fallback_alternative_plan(item, info_type)
        
        self.get_page_content(process_content)
    
    def fallback_alternative_plan(self, item, info_type):
        """Fallback method for generating an alternative plan when LLM fails"""
        self.add_action_log(f"Fallback alternative plan for: {item} ({info_type})")
        
        # Simple search approach
        search_query = f"{item} {info_type}"
        search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
        
        plan = {
            "strategy": "search_engine",
            "steps": [
                {
                    "action": "navigate",
                    "description": "Navigate to Google search",
                    "parameters": {
                        "url": search_url
                    }
                }
            ],
            "confidence": 0.7
        }
        
        self.add_action_log("Fallback Plan:")
        self.add_action_log(f"  Navigate to: {search_url}")
        
        # Execute the fallback plan
        self.execute_plan_step(plan)
    
    def get_page_content(self, callback):
        """Get the current page HTML content"""
        def handle_html(html):
            callback(html)
        self.web.page().toHtml(handle_html)
    
    def fill_form_with_llm(self, form_data):
        """Use LLM to understand form context and fill form fields"""
        self.add_action_log(f"Filling form with LLM: {form_data}")
        
        def process_content(html):
            # Parse HTML and extract form information
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract form elements
            forms = soup.find_all('form')
            form_info = []
            
            for i, form in enumerate(forms):
                form_data = {
                    "index": i,
                    "action": form.get('action', ''),
                    "method": form.get('method', 'GET'),
                    "fields": []
                }
                
                # Find input fields, textareas, and selects
                inputs = form.find_all(['input', 'textarea', 'select'])
                for input_elem in inputs:
                    field_info = {
                        "type": input_elem.get('type', 'text'),
                        "name": input_elem.get('name', ''),
                        "id": input_elem.get('id', ''),
                        "placeholder": input_elem.get('placeholder', ''),
                        "label": ""
                    }
                    
                    # Try to find associated label
                    if field_info["id"]:
                        label = soup.find('label', attrs={'for': field_info["id"]})
                        if label:
                            field_info["label"] = label.get_text().strip()
                    
                    form_data["fields"].append(field_info)
                
                form_info.append(form_data)
            
            # Use LLM to match form fields with provided data
            if form_info:
                forms_text = ""
                for form in form_info:
                    forms_text += f"Form {form['index']}:\n"
                    forms_text += f"  Action: {form['action']}\n"
                    forms_text += f"  Method: {form['method']}\n"
                    forms_text += "  Fields:\n"
                    for field in form["fields"]:
                        forms_text += f"    - Type: {field['type']}, Name: {field['name']}, ID: {field['id']}, Placeholder: {field['placeholder']}, Label: {field['label']}\n"
                    forms_text += "\n"
                
                prompt = f"""
Analyze the following web forms and match them with the provided data.
Determine which form and fields should be filled with which values.

Web forms:
{forms_text}

Data to fill:
{form_data}

Please provide your response in the following JSON format:
{{
  "form_index": "The index of the form to fill (or null if no suitable form found)",
  "field_mappings": [
    {{
      "field_name": "The name/id of the form field",
      "value": "The value to fill in this field"
    }}
  ],
  "confidence": "How confident you are in your form selection (0.0 to 1.0)"
}}

Just provide the JSON response, nothing else.
"""
                
                # Call the LLM API using centralized configuration
                api_endpoint = get_api_endpoint()
                headers = get_headers()
                
                payload = {
                    "model": get_model(),
                    "prompt": prompt,
                    "max_tokens": 600,
                    "temperature": 0.3,
                    "stream": False
                }
                
                try:
                    response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract the JSON response from the LLM
                    llm_response = result["choices"][0]["text"].strip()
                    
                    # Parse the JSON response
                    import json
                    parsed_response = json.loads(llm_response)
                    
                    form_index = parsed_response["form_index"]
                    field_mappings = parsed_response["field_mappings"]
                    confidence = parsed_response["confidence"]
                    
                    self.add_action_log(f"LLM Form Analysis - Form Index: {form_index}, Confidence: {confidence}")
                    
                    # If we have a form to fill and high confidence, proceed
                    if form_index is not None and confidence > 0.7:
                        self.add_action_log(f"Filling form {form_index} with {len(field_mappings)} fields")
                        # In a real implementation, we would use JavaScript execution to fill the form
                        # For now, we'll just log the action
                        for mapping in field_mappings:
                            self.add_action_log(f"  Filling field '{mapping['field_name']}' with '{mapping['value']}'")
                    else:
                        self.add_action_log("LLM could not confidently identify a form to fill")
                        
                except Exception as e:
                    self.add_action_log(f"Error in LLM form filling: {str(e)}")
            else:
                self.add_action_log("No forms found on the page")
        
        self.get_page_content(process_content)
    
    def click_element_with_llm(self, element_description):
        """Use LLM to identify and click on page elements"""
        self.add_action_log(f"Clicking element with LLM: {element_description}")
        
        def process_content(html):
            # Parse HTML and extract interactive elements
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract buttons, links, and other interactive elements
            interactive_elements = []
            
            # Find buttons
            buttons = soup.find_all('button')
            for button in buttons:
                element_info = {
                    "type": "button",
                    "text": button.get_text().strip(),
                    "id": button.get('id', ''),
                    "class": button.get('class', []),
                    "onclick": button.get('onclick', '')
                }
                interactive_elements.append(element_info)
            
            # Find links
            links = soup.find_all('a', href=True)
            for link in links:
                element_info = {
                    "type": "link",
                    "text": link.get_text().strip(),
                    "href": link.get('href', ''),
                    "id": link.get('id', ''),
                    "class": link.get('class', [])
                }
                interactive_elements.append(element_info)
            
            # Find input elements that might be clickable (e.g., submit buttons)
            inputs = soup.find_all('input', type=['submit', 'button'])
            for input_elem in inputs:
                element_info = {
                    "type": "input",
                    "text": input_elem.get('value', input_elem.get('placeholder', '')),
                    "id": input_elem.get('id', ''),
                    "class": input_elem.get('class', []),
                    "input_type": input_elem.get('type', '')
                }
                interactive_elements.append(element_info)
            
            # Use LLM to identify the most relevant element to click
            if interactive_elements:
                elements_text = ""
                for i, element in enumerate(interactive_elements[:20]):  # Limit to first 20 elements
                    elements_text += f"{i+1}. Type: {element['type']}, Text: {element['text'][:100]}\n"
                    if element['id']:
                        elements_text += f"   ID: {element['id']}\n"
                    if element['class']:
                        elements_text += f"   Class: {element['class']}\n"
                    elements_text += "\n"
                
                prompt = f"""
Analyze the following interactive elements on a web page and determine which one best matches the description: "{element_description}"

Interactive elements:
{elements_text}

Please provide your response as a single number indicating the index of the most relevant element to click.
For example, if element 3 is most relevant, just respond with "3".
If no element matches the description, respond with "0".
"""
                
                # Call the LLM API using centralized configuration
                api_endpoint = get_api_endpoint()
                headers = get_headers()
                
                payload = {
                    "model": get_model(),
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.3,
                    "stream": False
                }
                
                try:
                    response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract the response from the LLM
                    llm_response = result["choices"][0]["text"].strip()
                    
                    # Try to parse the response as an integer
                    try:
                        selected_index = int(llm_response) - 1  # Convert to 0-based index
                        if 0 <= selected_index < len(interactive_elements):
                            element = interactive_elements[selected_index]
                            self.add_action_log(f"LLM selected element to click: {element['text'][:100]}")
                            # In a real implementation, we would use JavaScript execution to click the element
                            # For now, we'll just log the action
                            self.add_action_log(f"  Would click element of type '{element['type']}'")
                        elif selected_index == -1:
                            self.add_action_log("LLM determined no element matches the description")
                        else:
                            self.add_action_log("LLM response out of range")
                    except ValueError:
                        self.add_action_log(f"LLM response not a valid number: {llm_response}")
                        
                except Exception as e:
                    self.add_action_log(f"Error in LLM element clicking: {str(e)}")
            else:
                self.add_action_log("No interactive elements found on the page")
        
        self.get_page_content(process_content)
    
    def search_for_element_with_llm(self, keyword):
        """Search for elements containing a keyword using LLM analysis"""
        self.add_action_log(f"Searching for: {keyword} using LLM")
        
        def process_content(html):
            # Parse HTML and search for keyword
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract text content and structure
            text_content = soup.get_text()
            
            # Use LLM to identify relevant elements
            prompt = f"""
Analyze the following web page content and identify elements related to "{keyword}".
Focus on finding relevant information, links, or interactive elements.

Web page content:
{text_content[:2000]}  # Limit content to 2000 characters

Please provide your response in the following JSON format:
{{
  "elements": [
    {{
      "type": "The type of element (e.g., 'text', 'link', 'button', 'form_field')",
      "content": "The relevant content or text",
      "selector": "A CSS selector or description of how to find this element (if applicable)",
      "relevance": "How relevant this element is to the keyword (0.0 to 1.0)"
    }}
  ],
  "summary": "A brief summary of what you found"
}}

Just provide the JSON response, nothing else.
"""
            
            # Call the LLM API using centralized configuration
            api_endpoint = get_api_endpoint()
            headers = get_headers()
            
            payload = {
                "model": get_model(),
                "prompt": prompt,
                "max_tokens": 800,
                "temperature": 0.3,
                "stream": False
            }
            
            try:
                response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # Extract the JSON response from the LLM
                llm_response = result["choices"][0]["text"].strip()
                
                # Parse the JSON response
                import json
                parsed_response = json.loads(llm_response)
                
                elements = parsed_response["elements"]
                summary = parsed_response["summary"]
                
                self.add_action_log(f"LLM Analysis Summary: {summary}")
                
                # Process the identified elements
                if elements:
                    self.add_action_log(f"Found {len(elements)} relevant elements:")
                    for i, element in enumerate(elements[:5]):  # Limit to first 5 elements
                        self.add_action_log(f"  {i+1}. {element['type']}: {element['content'][:100]}")
                else:
                    self.add_action_log("No relevant elements found by LLM")
                
                return elements
                
            except Exception as e:
                self.add_action_log(f"Error in LLM element search: {str(e)}")
                # Fall back to the original method
                return self.search_for_element_regex(keyword)
        
        self.get_page_content(process_content)
    
    def search_for_element_regex(self, keyword):
        """Original regex-based element search (fallback method)"""
        self.add_action_log(f"Searching for: {keyword} using regex (fallback)")
        
        def process_content(html):
            # Parse HTML and search for keyword
            soup = BeautifulSoup(html, 'html.parser')
            
            # First, try to find direct matches
            elements = soup.find_all(string=re.compile(keyword, re.I))  # Case insensitive search
            if elements:
                self.add_action_log(f"Found {len(elements)} direct matches for '{keyword}'")
                # Extract parent elements for context
                for element in elements[:3]:  # Limit to first 3 matches
                    parent = element.parent
                    if parent:
                        self.add_action_log(f"  Found in: {parent.name} - {str(parent)[:100]}...")
                return
            
            # If no direct matches, look for links that might be relevant
            links = soup.find_all('a', href=True)
            relevant_links = []
            for link in links:
                link_text = link.get_text().lower()
                if keyword.lower() in link_text or keyword.lower() in link.get('href', ''):
                    relevant_links.append(link)
            
            if relevant_links:
                self.add_action_log(f"Found {len(relevant_links)} potentially relevant links")
                for link in relevant_links[:3]:  # Limit to first 3 matches
                    href = link.get('href', '')
                    text = link.get_text()[:100]
                    self.add_action_log(f"  Link: {text} -> {href}")
            else:
                self.add_action_log(f"No elements or links found containing '{keyword}'")
        
        self.get_page_content(process_content)
    
    def search_for_element(self, keyword):
        """Search for elements containing a keyword (main method)"""
        # Use LLM-based element search as the primary method
        return self.search_for_element_with_llm(keyword)
    
    def check_requirements_fulfilled(self, item, info_type, extracted_info):
        """Check if the extracted information fulfills the user's requirements"""
        if not extracted_info:
            self.add_action_log(f"Unable to fulfill request for '{item}' - no {info_type} information found")
            return False
        
        self.add_action_log(f"Successfully fulfilled request for '{item}' - found {info_type} information:")
        # Limit the amount of information displayed
        if isinstance(extracted_info, list):
            for info in extracted_info[:3]:  # Show first 3 items
                self.add_action_log(f"  - {info}")
        else:
            self.add_action_log(f"  - {extracted_info}")
        
        return True
    
    def extract_information(self, info_type):
        """Extract specific information from the page"""
        self.add_action_log(f"Extracting information: {info_type}")
        
        def process_content(html):
            # Parse HTML and extract information
            soup = BeautifulSoup(html, 'html.parser')
            extracted_info = []
            
            if info_type == "price":
                # Look for price patterns
                price_patterns = [
                    re.compile(r'\$\d+(?:\.\d{2})?', re.IGNORECASE),
                    re.compile(r'\d+(?:\.\d{2})?\s*(?:USD|dollars)', re.IGNORECASE),
                    re.compile(r'€\d+(?:\.\d{2})?', re.IGNORECASE),
                    re.compile(r'£\d+(?:\.\d{2})?', re.IGNORECASE)
                ]
                
                for pattern in price_patterns:
                    matches = pattern.findall(html)
                    if matches:
                        extracted_info.extend(matches[:5])  # Limit to first 5 matches
                
                # Look for elements with common price classes/ids
                price_selectors = ['price', 'cost', 'amount', 'value', 'price-container', 'product-price']
                for selector in price_selectors:
                    elements = soup.find_all(attrs={'class': re.compile(selector, re.I)})
                    if elements:
                        for el in elements[:3]:
                            text = el.get_text(strip=True)
                            if text and text not in extracted_info:
                                extracted_info.append(text)
                    
                    elements = soup.find_all(attrs={'id': re.compile(selector, re.I)})
                    if elements:
                        for el in elements[:3]:
                            text = el.get_text(strip=True)
                            if text and text not in extracted_info:
                                extracted_info.append(text)
                
                if extracted_info:
                    self.add_action_log(f"Found prices: {', '.join(extracted_info[:5])}")
                    return self.check_requirements_fulfilled("item", "price", extracted_info)
                else:
                    self.add_action_log("No price information found")
                    return self.check_requirements_fulfilled("item", "price", None)
            
            elif info_type == "availability":
                # Look for availability information
                availability_patterns = [
                    re.compile(r'(in stock|out of stock|available|unavailable)', re.IGNORECASE),
                    re.compile(r'(\d+\s*items?\s*(?:left|remaining|in stock))', re.IGNORECASE)
                ]
                
                for pattern in availability_patterns:
                    matches = pattern.findall(html)
                    if matches:
                        extracted_info.extend(matches[:5])  # Limit to first 5 matches
                
                # Look for elements with common availability classes/ids
                availability_selectors = ['stock', 'availability', 'inventory', 'status']
                for selector in availability_selectors:
                    elements = soup.find_all(attrs={'class': re.compile(selector, re.I)})
                    if elements:
                        for el in elements[:3]:
                            text = el.get_text(strip=True)
                            if text and text not in extracted_info:
                                extracted_info.append(text)
                    
                    elements = soup.find_all(attrs={'id': re.compile(selector, re.I)})
                    if elements:
                        for el in elements[:3]:
                            text = el.get_text(strip=True)
                            if text and text not in extracted_info:
                                extracted_info.append(text)
                
                if extracted_info:
                    self.add_action_log(f"Found availability info: {', '.join(extracted_info[:5])}")
                    return self.check_requirements_fulfilled("item", "availability", extracted_info)
                else:
                    self.add_action_log("No availability information found")
                    return self.check_requirements_fulfilled("item", "availability", None)
            
            elif info_type == "reviews":
                # Look for review information
                review_patterns = [
                    re.compile(r'(\d+\s*reviews?)', re.IGNORECASE),
                    re.compile(r'(\d+\s*out of\s*\d+\s*stars?)', re.IGNORECASE),
                    re.compile(r'(rated\s*\d+(?:\.\d+)?)', re.IGNORECASE)
                ]
                
                for pattern in review_patterns:
                    matches = pattern.findall(html)
                    if matches:
                        extracted_info.extend(matches[:5])  # Limit to first 5 matches
                
                # Look for elements with common review classes/ids
                review_selectors = ['review', 'rating', 'stars', 'score']
                for selector in review_selectors:
                    elements = soup.find_all(attrs={'class': re.compile(selector, re.I)})
                    if elements:
                        for el in elements[:3]:
                            text = el.get_text(strip=True)
                            if text and text not in extracted_info:
                                extracted_info.append(text)
                    
                    elements = soup.find_all(attrs={'id': re.compile(selector, re.I)})
                    if elements:
                        for el in elements[:3]:
                            text = el.get_text(strip=True)
                            if text and text not in extracted_info:
                                extracted_info.append(text)
                
                if extracted_info:
                    self.add_action_log(f"Found review info: {', '.join(extracted_info[:5])}")
                    return self.check_requirements_fulfilled("item", "reviews", extracted_info)
                else:
                    self.add_action_log("No review information found")
                    return self.check_requirements_fulfilled("item", "reviews", None)
            
            elif info_type == "title":
                title = soup.title.string if soup.title else "No title found"
                self.add_action_log(f"Page title: {title}")
                return self.check_requirements_fulfilled("page", "title", title)
            
            else:
                # Generic information extraction - look for any structured data
                # Try to find any structured information on the page
                important_elements = []
                
                # Look for definition lists, tables, and other structured data
                dl_elements = soup.find_all('dl')
                table_elements = soup.find_all('table')
                list_elements = soup.find_all(['ul', 'ol'])
                
                if dl_elements:
                    self.add_action_log(f"Found {len(dl_elements)} definition lists with structured information")
                    for dl in dl_elements[:2]:
                        dt_elements = dl.find_all('dt')
                        for dt in dt_elements[:3]:
                            dd = dt.find_next_sibling('dd')
                            if dd:
                                info = f"{dt.get_text(strip=True)}: {dd.get_text(strip=True)[:100]}"
                                extracted_info.append(info)
                
                if table_elements:
                    self.add_action_log(f"Found {len(table_elements)} tables with structured information")
                    for table in table_elements[:2]:
                        rows = table.find_all('tr')
                        for row in rows[:3]:
                            cells = row.find_all(['td', 'th'])
                            if cells:
                                cell_texts = [cell.get_text(strip=True) for cell in cells]
                                info = ' | '.join(cell_texts[:100])
                                extracted_info.append(info)
                
                if not dl_elements and not table_elements:
                    # If no structured data, try to extract key information
                    # Look for headings and their following content
                    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if headings:
                        self.add_action_log(f"Found {len(headings)} headings, extracting content around them")
                        for heading in headings[:3]:
                            # Get the next few siblings
                            siblings = []
                            sibling = heading.next_sibling
                            count = 0
                            while sibling and count < 5:
                                if sibling.name and sibling.get_text(strip=True):
                                    siblings.append(sibling.get_text(strip=True)[:100])
                                    count += 1
                                sibling = sibling.next_sibling
                            
                            if siblings:
                                info = f"Content after '{heading.get_text(strip=True)}': {' ... '.join(siblings)}"
                                extracted_info.append(info)
                
                if extracted_info:
                    # Limit the amount of information displayed
                    display_info = extracted_info[:5]  # Show first 5 items
                    self.add_action_log(f"Found general information:")
                    for info in display_info:
                        self.add_action_log(f"  - {info}")
                    return self.check_requirements_fulfilled("item", "general", extracted_info)
                else:
                    self.add_action_log(f"No specific structured information found for '{info_type}'")
                    return self.check_requirements_fulfilled("item", "general", None)

        self.get_page_content(process_content)
    
    def wait_for_page_load(self, callback):
        """Wait for the page to finish loading before executing callback"""
        def check_load_finished():
            # In a real implementation, you would check if the page has finished loading
            # For now, we'll just execute the callback immediately
            callback()
        
        # Simulate waiting for page load
        QTimer.singleShot(2000, check_load_finished)  # Wait 2 seconds
    
    def determine_relevant_url_with_llm(self, item):
        """Use LLM to determine the most relevant URL for an item"""
        self.add_action_log(f"LLM analyzing item: {item}")
        
        # Create a prompt for the LLM to determine the most relevant domain
        prompt = f"""
Analyze the following item and determine the most relevant website domain to search for information about it:

Item: "{item}"

Please provide your response in the following JSON format:
{{
  "domain": "The most relevant domain URL (e.g., 'https://www.amazon.com', 'https://www.bestbuy.com')",
  "category": "The category of the item (e.g., 'electronics', 'travel', 'shopping', 'news', 'weather', 'books')",
  "confidence": "How confident you are in your domain selection (0.0 to 1.0)"
}}

Examples:
Item: "iPhone 17"
Response: {{"domain": "https://www.apple.com", "category": "electronics", "confidence": 0.95}}

Item: "flight to Paris"
Response: {{"domain": "https://www.google.com/flights", "category": "travel", "confidence": 0.9}}

Item: "bestseller book"
Response: {{"domain": "https://www.amazon.com", "category": "books", "confidence": 0.85}}

Just provide the JSON response, nothing else.
"""
        
        # Call the LLM API using centralized configuration
        api_endpoint = get_api_endpoint()
        headers = get_headers()
        
        payload = {
            "model": get_model(),
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.3,
            "stream": False
        }
        
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the JSON response from the LLM
            llm_response = result["choices"][0]["text"].strip()
            
            # Parse the JSON response
            import json
            parsed_response = json.loads(llm_response)
            
            domain = parsed_response["domain"]
            category = parsed_response["category"]
            confidence = parsed_response["confidence"]
            
            self.add_action_log(f"LLM Domain Analysis - Domain: {domain}, Category: {category}, Confidence: {confidence}")
            
            # If confidence is high enough, return the domain
            if confidence > 0.7:
                return domain
            else:
                self.add_action_log(f"Low confidence in LLM domain analysis ({confidence}), will use general search approach")
                return None
                
        except Exception as e:
            self.add_action_log(f"Error in LLM domain analysis: {str(e)}")
            # Fall back to the original method
            return self.determine_relevant_url_regex(item)
    
    def determine_relevant_url_regex(self, item):
        """Original regex-based domain determination (fallback method)"""
        # This method would ideally use a more sophisticated AI approach
        # For now, we'll implement a more flexible pattern matching system
        # that could be extended with machine learning in a real implementation
        
        item_lower = item.lower()
        self.add_action_log(f"AI analyzing item (regex fallback): {item}")
        
        # Define categories and their associated domains
        categories = {
            "electronics": {
                "keywords": ["phone", "iphone", "smartphone", "tablet", "ipad", "laptop", "computer", "macbook", "tv", "television", "headphones", "airpods", "speaker", "camera"],
                "domains": ["https://www.apple.com", "https://www.bestbuy.com", "https://www.amazon.com"]
            },
            "travel": {
                "keywords": ["flight", "airline", "ticket", "travel", "airport", "booking", "hotel"],
                "domains": ["https://www.google.com/flights", "https://www.expedia.com", "https://www.kayak.com"]
            },
            "shopping": {
                "keywords": ["buy", "purchase", "order", "product", "item", "store", "retail", "shop"],
                "domains": ["https://www.amazon.com", "https://www.target.com", "https://www.walmart.com", "https://www.bestbuy.com"]
            },
            "news": {
                "keywords": ["news", "headline", "article", "report", "current", "event", "story"],
                "domains": ["https://news.google.com", "https://www.bbc.com", "https://www.cnn.com"]
            },
            "weather": {
                "keywords": ["weather", "temperature", "forecast", "rain", "sun", "snow", "climate"],
                "domains": ["https://weather.com", "https://www.accuweather.com"]
            },
            "books": {
                "keywords": ["book", "novel", "author", "read", "literature", "ebook"],
                "domains": ["https://www.amazon.com", "https://www.barnesandnoble.com"]
            }
        }
        
        # Score each category based on keyword matches
        category_scores = {}
        for category, data in categories.items():
            score = 0
            for keyword in data["keywords"]:
                if keyword in item_lower:
                    score += 1
            category_scores[category] = score
        
        # Find the category with the highest score
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        # If we have a good match, return the first domain for that category
        if best_score > 0:
            best_domain = categories[best_category]["domains"][0]
            self.add_action_log(f"AI determined best domain for '{item}': {best_domain} (category: {best_category}, score: {best_score})")
            return best_domain
        
        # For a more advanced implementation, we could:
        # 1. Use a neural network to predict the most relevant domain
        # 2. Learn from user interactions and feedback
        # 3. Use semantic analysis to understand the context better
        # 4. Query an API that specializes in domain prediction
        
        self.add_action_log(f"AI could not determine a specific domain for '{item}', will use search approach")
        return None
    
    def determine_relevant_url(self, item):
        """Use AI to determine the most relevant URL for an item (main method)"""
        # Use LLM-based domain determination as the primary method
        return self.determine_relevant_url_with_llm(item)
    
    def create_execution_plan_with_llm(self, item, info_type):
        """Create an execution plan using LLM reasoning and adaptive strategies"""
        self.add_action_log(f"Creating execution plan for: {item} ({info_type}) using LLM")
        
        
        def process_content(html):
            # Extract text content for LLM analysis
            soup = BeautifulSoup(html, 'html.parser')
            text_content = soup.get_text()[:2000]  # Limit content to 2000 characters
            
            # Use LLM to create a more sophisticated execution plan
            prompt = f"""
Analyze the current web page content and create an execution plan to find information about "{item}" of type "{info_type}".

Current page content:
{text_content}

Please provide your response in the following JSON format:
{{
  "strategy": "The overall strategy to use (e.g., 'direct_navigation', 'search_engine', 'form_submission', 'multi_step_process')",
  "steps": [
    {{
      "action": "The action to take (e.g., 'navigate', 'search', 'fill_form', 'click', 'extract')",
      "description": "A detailed description of what to do",
      "parameters": {{
        "url": "URL to navigate to (if applicable)",
        "query": "Search query (if applicable)",
        "form_data": {{}},  # Form data to fill (if applicable)
        "element_description": "Description of element to click (if applicable)",
        "keywords": "Keywords to search for (if applicable)"
      }}
    }}
  ],
  "confidence": "How confident you are in this plan (0.0 to 1.0)"
}}

Examples:
For finding the price of an iPhone on Apple's website:
{{
  "strategy": "direct_navigation",
  "steps": [
    {{
      "action": "navigate",
      "description": "Go to Apple's website",
      "parameters": {{
        "url": "https://www.apple.com"
      }}
    }},
    {{
      "action": "search",
      "description": "Search for iPhone on the page",
      "parameters": {{
        "keywords": "iPhone"
      }}
    }},
    {{
      "action": "click",
      "description": "Click on the iPhone link",
      "parameters": {{
        "element_description": "Link to iPhone product page"
      }}
    }},
    {{
      "action": "extract",
      "description": "Extract price information",
      "parameters": {{
        "keywords": "price"
      }}
    }}
  ],
  "confidence": 0.9
}}

Just provide the JSON response, nothing else.
"""
            
            # Call the LLM API using centralized configuration
            api_endpoint = get_api_endpoint()
            headers = get_headers()
            
            payload = {
                "model": get_model(),
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.3,
                "stream": False
            }
            
            try:
                response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # Extract the JSON response from the LLM
                llm_response = result["choices"][0]["text"].strip()
                
                # Parse the JSON response
                import json
                parsed_response = json.loads(llm_response)
                
                strategy = parsed_response["strategy"]
                steps = parsed_response["steps"]
                confidence = parsed_response["confidence"]
                
                self.add_action_log(f"LLM Execution Plan - Strategy: {strategy}, Confidence: {confidence}")
                
                # Create a plan based on the LLM response
                plan = {
                    "strategy": strategy,
                    "steps": [],
                    "confidence": confidence
                }
                
                # Process each step
                for i, step in enumerate(steps):
                    action = step["action"]
                    description = step["description"]
                    parameters = step["parameters"]
                    
                    # Add step to plan
                    plan["steps"].append({
                        "action": action,
                        "description": description,
                        "parameters": parameters
                    })
                    
                    self.add_action_log(f"  Step {i+1}: {action} - {description}")
                
                return plan
                
            except Exception as e:
                self.add_action_log(f"Error in LLM execution planning: {str(e)}")
                # Fall back to the original method
                return self.create_execution_plan_regex(item, info_type)
        
        # Get current page content for LLM analysis
        self.get_page_content(process_content)
        # Return a default plan in case the async call doesn't complete immediately
        return self.create_execution_plan_regex(item, info_type)
    
    def create_execution_plan_regex(self, item, info_type):
        """Original regex-based execution plan creation (fallback method)"""
        self.add_action_log(f"Creating execution plan for: {item} ({info_type}) using regex (fallback)")
        
        # Use AI to determine if we have a relevant direct URL
        direct_url = self.determine_relevant_url(item)
        
        if direct_url:
            self.add_action_log(f"Plan: Direct navigation to AI-determined URL: {direct_url}")
            plan = {
                "steps": [
                    f"Navigate to {direct_url}",
                    f"Search for '{item}' on the page",
                    f"Click on the link for '{item}' to navigate to the details page",
                    f"Extract {info_type} information",
                ],
                "url": direct_url,
                "type": "direct"
            }
        else:
            # Need to search first
            self.add_action_log("Plan: Search engine approach - will search and then analyze results")
            search_query = f"{item}"
            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            
            plan = {
                "steps": [
                    f"Search for '{item}' on Google ({search_url})",
                    "Analyze search results and select most relevant link",
                    "Navigate to selected page",
                    f"Search for '{item}' on the page",
                    f"Extract {info_type} information"
                ],
                "url": search_url,
                "type": "search"
            }
        
        # Display the plan
        self.add_action_log("Execution Plan:")
        for i, step in enumerate(plan["steps"], 1):
            self.add_action_log(f"  {i}. {step}")
        
        return plan
    
    def create_execution_plan(self, item, info_type):
        """Create an execution plan based on the item and requested information (main method)"""
        # Use LLM-based execution planning as the primary method
        return self.create_execution_plan_with_llm(item, info_type)
    
    def analyze_search_results_with_llm(self, item, callback):
        """Analyze search results using LLM and select the most relevant link"""
        self.add_action_log(f"Analyzing search results for: {item} using LLM")
        
        def process_content(html):
            # Parse HTML and find search result links
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for search result links (Google-specific selectors)
            # This would need to be adapted for other search engines
            search_results = soup.find_all('a', href=True)
            
            # Filter for actual search result links (not navigation, ads, etc.)
            relevant_links = []
            for link in search_results:
                href = link.get('href', '')
                # Skip Google navigation links
                if href.startswith('/url?q='):
                    # Extract actual URL from Google's redirect
                    match = re.search(r'/url\?q=([^&]+)', href)
                    if match:
                        href = match.group(1)
                
                # Skip non-web URLs
                if href.startswith('http') and 'google' not in href:
                    text = link.get_text()
                    title = link.find_previous('h3')
                    title_text = title.get_text() if title else ''
                    
                    # Collect relevant links with their text content
                    relevant_links.append({
                        'url': href,
                        'text': text[:200],  # Limit text length
                        'title': title_text[:200] if title_text else ''  # Limit title length
                    })
            
            # If we have relevant links, use LLM to rank them
            if relevant_links:
                # Create a prompt for the LLM to rank the search results
                links_text = ""
                for i, link in enumerate(relevant_links[:10]):  # Limit to first 10 links
                    links_text += f"{i+1}. Title: {link['title']}\n   Text: {link['text']}\n   URL: {link['url']}\n\n"
                
                prompt = f"""
Analyze the following search results for the query "{item}" and rank them by relevance.
Return only the index (1-{min(10, len(relevant_links))}) of the most relevant result.

Search Results:
{links_text}

Please provide your response as a single number indicating the index of the most relevant result.
For example, if result 3 is most relevant, just respond with "3".
"""
                
                # Call the LLM API using centralized configuration
                api_endpoint = get_api_endpoint()
                headers = get_headers()
                
                payload = {
                    "model": get_model(),
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.3,
                    "stream": False
                }
                
                try:
                    response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract the response from the LLM
                    llm_response = result["choices"][0]["text"].strip()
                    
                    # Try to parse the response as an integer
                    try:
                        selected_index = int(llm_response) - 1  # Convert to 0-based index
                        if 0 <= selected_index < len(relevant_links):
                            best_link = relevant_links[selected_index]
                            self.add_action_log(f"LLM selected most relevant link: {best_link['title']}")
                            self.add_action_log(f"URL: {best_link['url']}")
                            callback(best_link['url'])
                        else:
                            # If LLM response is out of range, fall back to the first link
                            self.add_action_log(f"LLM response out of range, falling back to first link")
                            best_link = relevant_links[0]
                            self.add_action_log(f"Selected first link: {best_link['title']}")
                            self.add_action_log(f"URL: {best_link['url']}")
                            callback(best_link['url'])
                    except ValueError:
                        # If LLM response is not a valid integer, fall back to the first link
                        self.add_action_log(f"LLM response not a valid number, falling back to first link")
                        best_link = relevant_links[0]
                        self.add_action_log(f"Selected first link: {best_link['title']}")
                        self.add_action_log(f"URL: {best_link['url']}")
                        callback(best_link['url'])
                        
                except Exception as e:
                    self.add_action_log(f"Error in LLM search result analysis: {str(e)}")
                    # Fall back to the original method
                    return self.analyze_search_results_regex(item, callback)
            else:
                self.add_action_log("No relevant links found in search results")
                # Fallback to the current page
                callback(None)
        
        self.get_page_content(process_content)
    
    def analyze_search_results_regex(self, item, callback):
        """Original regex-based search result analysis (fallback method)"""
        self.add_action_log(f"Analyzing search results for: {item} using regex (fallback)")
        
        def process_content(html):
            # Parse HTML and find search result links
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for search result links (Google-specific selectors)
            # This would need to be adapted for other search engines
            search_results = soup.find_all('a', href=True)
            
            # Filter for actual search result links (not navigation, ads, etc.)
            relevant_links = []
            for link in search_results:
                href = link.get('href', '')
                # Skip Google navigation links
                if href.startswith('/url?q='):
                    # Extract actual URL from Google's redirect
                    match = re.search(r'/url\?q=([^&]+)', href)
                    if match:
                        href = match.group(1)
                
                # Skip non-web URLs
                if href.startswith('http') and 'google' not in href:
                    # Score the link based on relevance
                    text = link.get_text().lower()
                    title = link.find_previous('h3')
                    title_text = title.get_text().lower() if title else ''
                    
                    # Calculate relevance score
                    score = 0
                    item_lower = item.lower()
                    
                    # Score based on exact matches
                    if item_lower in text:
                        score += 3
                    if item_lower in title_text:
                        score += 5
                    
                    # Score based on partial matches
                    item_words = item_lower.split()
                    for word in item_words:
                        if len(word) > 3:  # Only consider words longer than 3 characters
                            if word in text:
                                score += 1
                            if word in title_text:
                                score += 2
                    
                    if score > 0:
                        relevant_links.append({
                            'url': href,
                            'text': link.get_text()[:100],
                            'title': title_text[:100] if title_text else '',
                            'score': score
                        })
            
            # Sort by score and select the best link
            if relevant_links:
                relevant_links.sort(key=lambda x: x['score'], reverse=True)
                best_link = relevant_links[0]
                self.add_action_log(f"Selected most relevant link (score: {best_link['score']}): {best_link['text']}")
                self.add_action_log(f"URL: {best_link['url']}")
                callback(best_link['url'])
            else:
                self.add_action_log("No relevant links found in search results")
                # Fallback to the current page
                callback(None)
        
        self.get_page_content(process_content)
    
    def analyze_search_results(self, item, callback):
        """Analyze search results and select the most relevant link (main method)"""
        # Use LLM-based search result analysis as the primary method
        return self.analyze_search_results_with_llm(item, callback)
    
    def execute_plan_step(self, plan, step_index=0):
        """Execute a step of the plan"""
        # Check if plan is valid
        if not plan or "steps" not in plan:
            self.add_action_log("Invalid plan structure, cannot execute")
            return
            
        if step_index >= len(plan["steps"]):
            self.add_action_log("Plan execution completed")
            return
        
        step = plan["steps"][step_index]
        self.add_action_log(f"Executing step {step_index + 1}: {step}")
        
        # Handle different types of steps
        if isinstance(step, str):
            # Handle string-based steps (from regex-based planning)
            if step.startswith("Navigate to"):
                url = plan.get("url", "")
                if url:
                    self.navigate_to_url(url)
                    # After navigation, continue with next step
                    QTimer.singleShot(3000, lambda: self.execute_plan_step(plan, step_index + 1))
                else:
                    self.add_action_log("No URL found for navigation step")
                    QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
            elif step.startswith("Search for"):
                # Extract item name from the step
                item_match = re.search(r"'(.+?)'", step)
                if item_match:
                    item = item_match.group(1)
                    self.search_for_element(item)
                QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
            elif step.startswith("Click on the link"):
                # Extract item name from the step
                item_match = re.search(r"'(.+?)'", step)
                if item_match:
                    item = item_match.group(1)
                    self.click_element_with_llm(item)
                QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
            elif step.startswith("Extract"):
                # Extract info type from the step
                info_match = re.search(r"Extract (.+?) information", step)
                if info_match:
                    info_type = info_match.group(1)
                    self.extract_information(info_type)
                QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
            elif step.startswith("Analyze search results"):
                # Extract item name from the plan
                if len(plan["steps"]) > 0:
                    search_step = plan["steps"][0]  # Get the search step
                    item_match = re.search(r"'(.+?)'", search_step)
                    if item_match:
                        item = item_match.group(1)
                        
                        def on_link_selected(url):
                            if url:
                                self.add_action_log(f"Navigating to selected link: {url}")
                                self.navigate_to_url(url)
                                # Continue with next steps after navigation
                                QTimer.singleShot(3000, lambda: self.execute_plan_step(plan, step_index + 1))
                            else:
                                # If no URL selected, continue with next step
                                QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
                        
                        self.analyze_search_results(item, on_link_selected)
                    else:
                        # If we can't extract item, continue with next step
                        QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
                else:
                    # If no steps, continue with next step
                    QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
            else:
                # For any other step, just continue
                QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
        elif isinstance(step, dict):
            # Handle dictionary-based steps (from LLM-based planning)
            action = step.get("action", "")
            parameters = step.get("parameters", {})
            
            if action == "navigate":
                url = parameters.get("url", "")
                if url:
                    # For navigation steps, we don't immediately continue with the next step
                    # Instead, we let the page analysis determine what to do next
                    self.navigate_to_url(url)
                    # We don't schedule the next step here because page analysis will handle it
                    return
                else:
                    self.add_action_log("No URL found for navigation step")
                    QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
            elif action == "search":
                keywords = parameters.get("keywords", "")
                if keywords:
                    self.search_for_element(keywords)
                QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
            elif action == "click":
                element_description = parameters.get("element_description", "")
                if element_description:
                    self.click_element_with_llm(element_description)
                QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
            elif action == "extract":
                keywords = parameters.get("keywords", "")
                if keywords:
                    self.extract_information(keywords)
                QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
            elif action == "fill_form":
                form_data = parameters.get("form_data", {})
                if form_data:
                    self.fill_form_with_llm(form_data)
                QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
            else:
                # For any other action, just continue
                QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
        else:
            # For any other step type, just continue
            QTimer.singleShot(2000, lambda: self.execute_plan_step(plan, step_index + 1))
    
    def parse_ai_command_with_llm(self, command):
        """Parse natural language command using LLM and convert to actions"""
        self.add_action_log(f"Parsing command with LLM: {command}")
        
        # Create a prompt for the LLM to extract intent and entities
        prompt = f"""
Analyze the following user command and extract the intent and entities:

Command: "{command}"

Please provide your response in the following JSON format:
{{
  "intent": "The user's intent (e.g., 'find_price', 'check_availability', 'get_reviews', 'search_news', 'find_flight', 'general_query')",
  "entities": {{
    "item": "The item or subject the user is asking about (if applicable)",
    "info_type": "The type of information requested (e.g., 'price', 'availability', 'reviews', 'general')"
  }},
  "confidence": "How confident you are in your analysis (0.0 to 1.0)"
}}

Examples:
Command: "What is the price of iPhone 17?"
Response: {{"intent": "find_price", "entities": {{"item": "iPhone 17", "info_type": "price"}}, "confidence": 0.95}}

Command: "Check if the MacBook Pro is available"
Response: {{"intent": "check_availability", "entities": {{"item": "MacBook Pro", "info_type": "availability"}}, "confidence": 0.9}}

Command: "Find reviews for the new Tesla Model S"
Response: {{"intent": "get_reviews", "entities": {{"item": "Tesla Model S", "info_type": "reviews"}}, "confidence": 0.85}}

Command: "What's in the news today?"
Response: {{"intent": "search_news", "entities": {{"item": "news", "info_type": "general"}}, "confidence": 0.8}}

Command: "Find cheap flights to Paris"
Response: {{"intent": "find_flight", "entities": {{"item": "flights to Paris", "info_type": "price"}}, "confidence": 0.9}}

Just provide the JSON response, nothing else.
"""
        
        # Call the LLM API using centralized configuration
        api_endpoint = get_api_endpoint()
        headers = get_headers()
        
        payload = {
            "model": get_model(),
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.3,
            "stream": False
        }
        
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the JSON response from the LLM
            llm_response = result["choices"][0]["text"].strip()
            
            # Parse the JSON response
            import json
            parsed_response = json.loads(llm_response)
            
            intent = parsed_response["intent"]
            entities = parsed_response["entities"]
            confidence = parsed_response["confidence"]
            
            self.add_action_log(f"LLM Analysis - Intent: {intent}, Item: {entities.get('item', 'N/A')}, Info Type: {entities.get('info_type', 'N/A')}, Confidence: {confidence}")
            
            # Set current request information for intelligent navigation
            self.current_request = entities
            self.request_fulfilled = False
            
            # If confidence is low, fall back to a general search
            if confidence < 0.5:
                self.add_action_log("Low confidence in LLM analysis, falling back to general search")
                plan = self.create_execution_plan(command, "general")
                self.execute_plan_step(plan)
                return True
            
            # Handle different intents
            if intent in ["find_price", "check_availability", "get_reviews"]:
                item = entities.get("item", "")
                info_type = entities.get("info_type", "general")
                if item:
                    plan = self.create_execution_plan(item, info_type)
                    self.execute_plan_step(plan)
                else:
                    # If no item identified, do a general search
                    plan = self.create_execution_plan(command, info_type)
                    self.execute_plan_step(plan)
                return True
            elif intent == "search_news":
                plan = self.create_execution_plan("news", "general")
                self.execute_plan_step(plan)
                return True
            elif intent == "find_flight":
                item = entities.get("item", "flights")
                plan = self.create_execution_plan(item, "price")
                self.execute_plan_step(plan)
                return True
            elif intent == "general_query":
                # For general queries, we'll search and then use the chat functionality
                plan = self.create_execution_plan(command, "general")
                self.execute_plan_step(plan)
                return True
            else:
                # Default action for unrecognized intents
                self.add_action_log(f"Unrecognized intent: {intent}, falling back to general search")
                plan = self.create_execution_plan(command, "general")
                self.execute_plan_step(plan)
                return True
                
        except Exception as e:
            self.add_action_log(f"Error in LLM parsing: {str(e)}")
            # Fall back to the original regex-based parsing
            return self.parse_ai_command_regex(command)
    
    def parse_ai_command_regex(self, command):
        """Original regex-based parsing (fallback method)"""
        command = command.strip()  # Keep original case for item names
        self.add_action_log(f"Parsing command with regex (fallback): {command}")
        
        # Convert to lowercase for pattern matching
        command_lower = command.lower()
        
        # Extract what the user wants to find
        # Look for patterns like "check the [item] price" or "find [item]"
        item_match = re.search(r'(?:check|find|look for|search for|get|show me)\s+(?:the\s+)?(.+?)(?:\s+(?:price|cost|availability|info|information|reviews|rating))?', command_lower)
        if not item_match:
            # Try a more general pattern
            item_match = re.search(r'(?:what is|what\'?s|how much is|how much does)\s+(?:the\s+)?(.+?)(?:\s+cost|\s+price)?', command_lower)
        if not item_match:
            # Try pattern for flight tickets or similar
            item_match = re.search(r'(?:search for|find|book)\s+(.+)', command_lower)
        
        if item_match:
            item = item_match.group(1).strip()
            # Remove trailing words like "for me", "please", etc.
            item = re.sub(r'\s+(for me|please|thanks|thank you)$', '', item, flags=re.IGNORECASE)
            self.add_action_log(f"Identified item: {item}")
            
            # Determine what information is requested
            info_type = "general"
            if "price" in command_lower or "cost" in command_lower:
                info_type = "price"
            elif "availability" in command_lower or "in stock" in command_lower or "available" in command_lower:
                info_type = "availability"
            elif "review" in command_lower or "rating" in command_lower:
                info_type = "reviews"
            elif "flight" in command_lower or "ticket" in command_lower or "airline" in command_lower:
                info_type = "price"  # For flights, we're typically interested in price
            
            self.add_action_log(f"Requested information type: {info_type}")
            
            # Create and execute plan
            plan = self.create_execution_plan(item, info_type)
            self.execute_plan_step(plan)
            return True
        
        # Handle other specific commands
        if "news" in command_lower:
            plan = self.create_execution_plan("news", "general")
            self.execute_plan_step(plan)
            return True
        elif "weather" in command_lower:
            plan = self.create_execution_plan("weather", "general")
            self.execute_plan_step(plan)
            return True
        elif "flight" in command_lower or "airline" in command_lower:
            # For flight searches, we might want to go to a specific flight search site
            plan = self.create_execution_plan("flight", "price")
            self.execute_plan_step(plan)
            return True
        
        # Default action if command not recognized
        self.add_action_log(f"Command not recognized: {command}")
        return False
    
    def parse_ai_command(self, command):
        """Parse natural language command and convert to actions (main method)"""
        # Use LLM-based parsing as the primary method
        return self.parse_ai_command_with_llm(command)
    
    def execute_ai_command(self):
        """Execute the AI command entered by the user"""
        command = self.ai_command.toPlainText().strip()
        if not command:
            self.add_action_log("Please enter a command first.")
            return
        
        self.add_action_log(f"Executing command: {command}")
        self.parse_ai_command(command)
    
    def ask_ai(self):
        # Capture the user's question immediately
        question = self.chat.toPlainText().strip()
        
        # Check if question is empty
        if not question:
            self.chat.append("Please enter a question first.")
            return
        
        # Store the question and clear the chat area for the response
        user_question = question
        self.chat.clear()
        
        # Show the user's question in the chat
        # Convert to HTML and insert
        user_question_html = f"<b>You:</b> {user_question}<br>"
        self.chat.insertHtml(user_question_html)
        ai_prompt_html = "<b>AI:</b> "
        self.chat.insertHtml(ai_prompt_html)
        
        # Get the current HTML content
        def handle_html(html):
            # Parse HTML and extract body content
            soup = BeautifulSoup(html, 'html.parser')
            body_content = soup.body.get_text(strip=False) if soup.body else html
            
            # Create a default prompt template
            template = "Based on the following web page content, please answer the question according to the content:\n\n{html}\n\nQuestion: {question}\n\nAnswer:"
            # Replace placeholders in template
            prompt = template.replace("{html}", body_content).replace("{question}", user_question)
            
            print("Prompt sent to AI:", prompt)  # Debugging line
            
            # Replace with your OpenAI compatible API endpoint and key
            api_endpoint = get_api_endpoint()  # Change this to your endpoint
            headers = get_headers()
            
            # For OpenAI completions API with streaming
            payload = {
                "model": get_model(),  # Change model as needed
                "prompt": prompt,
                "max_tokens": 131072,
                "temperature": 0.7,
                "stream": True  # Enable streaming
            }
            
            try:
                # Use streaming to get response in chunks
                response = requests.post(api_endpoint, headers=headers, json=payload, stream=True, timeout=120)
                response.raise_for_status()
                
                full_response = ""
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data = decoded_line[6:]  # Remove "data: " prefix
                            if data != "[DONE]":
                                try:
                                    json_data = json.loads(data)
                                    if "choices" in json_data and len(json_data["choices"]) > 0:
                                        if "text" in json_data["choices"][0]:
                                            chunk = json_data["choices"][0]["text"]
                                            full_response += chunk
                                            # Update the chat in real-time
                                            # Convert markdown to HTML and insert
                                            html_chunk = chunk.replace('\n', '<br>')
                                            self.chat.insertHtml(html_chunk)
                                            QApplication.processEvents()  # Ensure UI updates
                                except json.JSONDecodeError:
                                    continue
                
                if not full_response:
                    no_response_html = "\nNo response from AI".replace('\n', '<br>')
                    self.chat.insertHtml(no_response_html)
                    
            except requests.exceptions.RequestException as e:
                error_html = f"\nError contacting AI service: {str(e)}".replace('\n', '<br>')
                self.chat.insertHtml(error_html)
            except Exception as e:
                error_html = f"\nUnexpected error: {str(e)}".replace('\n', '<br>')
                self.chat.insertHtml(error_html)
        
        self.web.page().toHtml(handle_html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    browser = AIBrowser()
    browser.show()
    sys.exit(app.exec_())
