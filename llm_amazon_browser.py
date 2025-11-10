import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QLabel, 
                             QStatusBar, QProgressBar, QTextEdit, QSplitter, 
                             QListWidget, QGroupBox)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt, QDateTime, QTimer
import urllib.parse
import requests
import json
import re

# Import LLM configuration
from llm_config import get_llm_config, get_api_endpoint, get_api_key, get_model, get_headers

class LLMAmazonBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metis Browser")
        self.resize(1600, 1000)
        
        # Initialize conversation history
        self.conversation_history = []
        
        # LLM Configuration
        self.llm_config = get_llm_config()
        self.api_endpoint = get_api_endpoint()
        self.api_key = get_api_key()
        self.model = get_model()
        self.headers = get_headers()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Create address bar
        self.address_bar = QLineEdit()
        self.address_bar.setPlaceholderText("Enter URL (e.g., https://www.amazon.com)...")
        self.address_bar.returnPressed.connect(self.navigate_to_url)
        
        # Create navigation buttons
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self.go_back)
        
        self.forward_btn = QPushButton("Forward")
        self.forward_btn.clicked.connect(self.go_forward)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_page)
        
        self.home_btn = QPushButton("Amazon Home")
        self.home_btn.clicked.connect(self.go_home)
        
        self.go_btn = QPushButton("Go")
        self.go_btn.clicked.connect(self.navigate_to_url)
        
        # Create AI command interface
        self.ai_command = QTextEdit()
        self.ai_command.setPlaceholderText("Enter natural language commands (e.g., 'Find me a Bluetooth speaker under $50' or 'Show me the reviews for this product')...")
        self.ai_command.setMaximumHeight(80)
        
        self.ai_execute_btn = QPushButton("Execute AI Command")
        self.ai_execute_btn.clicked.connect(self.execute_ai_command)
        
        # Create AI actions log
        self.actions_log = QListWidget()
        self.actions_log.setMaximumHeight(200)
        
        # Create web view
        self.web = QWebEngineView()
        self.web.loadStarted.connect(self.load_started)
        self.web.loadProgress.connect(self.load_progress)
        self.web.loadFinished.connect(self.load_finished)
        self.web.urlChanged.connect(self.update_url_bar)
        
        # Timer for auto-navigation
        self.auto_nav_timer = QTimer()
        self.auto_nav_timer.timeout.connect(self.auto_navigate_to_best_item)
        self.pending_search_query = None
        self.search_results_page = False
        
        # Set initial URL to Amazon
        self.web.load(QUrl("https://www.google.com"))
        
        # Create AI Command Panel
        ai_panel = QGroupBox("AI Command Interface")
        ai_layout = QVBoxLayout()
        ai_layout.addWidget(self.ai_command)
        ai_layout.addWidget(self.ai_execute_btn)
        ai_layout.addWidget(QLabel("AI Actions Log:"))
        ai_layout.addWidget(self.actions_log)
        ai_panel.setLayout(ai_layout)
        ai_panel.setMaximumWidth(400)
        
        # Create browser controls panel
        browser_controls = QWidget()
        browser_layout = QVBoxLayout()
        
        # Layout for address bar
        address_layout = QHBoxLayout()
        address_layout.addWidget(QLabel("Address:"))
        address_layout.addWidget(self.address_bar)
        address_layout.addWidget(self.go_btn)
        
        # Layout for navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.back_btn)
        nav_layout.addWidget(self.forward_btn)
        nav_layout.addWidget(self.refresh_btn)
        nav_layout.addWidget(self.home_btn)
        nav_layout.addStretch()
        
        # Add components to browser layout
        browser_layout.addLayout(address_layout)
        browser_layout.addLayout(nav_layout)
        browser_layout.addWidget(self.web)
        browser_controls.setLayout(browser_layout)
        
        # Create splitter for main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(browser_controls)
        splitter.addWidget(ai_panel)
        
        # Set splitter as central widget
        self.setCentralWidget(splitter)
        
        # Set initial sizes for splitter
        splitter.setSizes([1200, 400])
        
        # Update URL bar with initial URL
        self.update_url_bar(QUrl("https://www.amazon.com"))
    
    def add_action_log(self, action):
        """Add an action to the actions log"""
        self.actions_log.addItem(f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] {action}")
        self.actions_log.scrollToBottom()
        
        # Add to conversation history
        self.conversation_history.append({
            "type": "action",
            "content": action,
            "timestamp": QDateTime.currentDateTime().toString()
        })
    
    def navigate_to_url(self):
        """Navigate to the URL entered in the address bar"""
        url_text = self.address_bar.text().strip()
        if url_text:
            # Check if it's a full URL
            if not url_text.startswith(('http://', 'https://')):
                url_text = 'https://' + url_text
            self.web.load(QUrl(url_text))
            self.status_bar.showMessage(f"Navigating to: {url_text}")
            self.add_action_log(f"Navigating to: {url_text}")
    
    def go_back(self):
        """Navigate back in browser history"""
        self.web.back()
        self.status_bar.showMessage("Going back...")
        self.add_action_log("Navigating back")
    
    def go_forward(self):
        """Navigate forward in browser history"""
        self.web.forward()
        self.status_bar.showMessage("Going forward...")
        self.add_action_log("Navigating forward")
    
    def refresh_page(self):
        """Refresh the current page"""
        self.web.reload()
        self.status_bar.showMessage("Refreshing page...")
        self.add_action_log("Refreshing page")
    
    def go_home(self):
        """Go to Amazon homepage"""
        self.web.load(QUrl("https://www.amazon.com"))
        self.status_bar.showMessage("Going to Amazon homepage...")
        self.address_bar.setText("https://www.amazon.com")
        self.add_action_log("Returning to Amazon homepage")
    
    def load_started(self):
        """Handle page load start"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Loading page...")
    
    def load_progress(self, progress):
        """Update progress bar"""
        self.progress_bar.setValue(progress)
    
    def load_finished(self, success):
        """Handle page load finish"""
        self.progress_bar.setVisible(False)
        if success:
            self.status_bar.showMessage("Page loaded successfully")
            # Check if we're on a search results page and need to auto-navigate
            current_url = self.web.url().toString()
            if "amazon.com/s?" in current_url and self.pending_search_query:
                self.search_results_page = True
                self.auto_nav_timer.start(3000)  # Start timer for 3 seconds
            else:
                self.search_results_page = False
        else:
            self.status_bar.showMessage("Failed to load page")
    
    def update_url_bar(self, url):
        """Update the URL bar with current URL"""
        self.address_bar.setText(url.toString())

    def auto_navigate_to_best_item(self):
        """Auto-navigate to the best matching item after search"""
        self.auto_nav_timer.stop()
        if self.pending_search_query and self.search_results_page:
            self.add_action_log(f"Auto-navigating to best item for: {self.pending_search_query}")
            
            # Use JavaScript to find and click the first search result
            js_code = """
            // Find the first search result link and click it
            var firstResult = document.querySelector('[data-component-type="s-search-result"] h2 a');
            if (!firstResult) {
                // Try alternative selectors
                firstResult = document.querySelector('.s-result-item h2 a');
            }
            if (!firstResult) {
                // Try another alternative
                firstResult = document.querySelector('[data-component-type="s-search-result"] .a-link-normal');
            }
            if (firstResult) {
                firstResult.click();
                "Clicked on first search result";
            } else {
                "No search results found";
            }
            """
            
            self.web.page().runJavaScript(js_code, self.js_callback)
            self.pending_search_query = None
            self.search_results_page = False
    
    def js_callback(self, result):
        """Callback for JavaScript execution"""
        if result:
            self.add_action_log(f"JavaScript result: {result}")

    def interpret_user_input_with_llm(self, user_input):
        """Use LLM to interpret user input and determine actions"""
        self.add_action_log(f"Interpreting user input with LLM: {user_input}")
        
        # Create a prompt for the LLM to determine the intent and actions
        prompt = f"""
You are an AI assistant for an Amazon browser. Analyze the following user command and determine what actions the browser should take.

User Command: "{user_input}"

Please provide your response in the following JSON format:
{{
  "intent": "The user's intent (e.g., 'search_products', 'filter_results', 'extract_info', 'compare_products', 'navigate')",
  "actions": [
    {{
      "type": "The type of action (e.g., 'search_amazon', 'navigate_to_url', 'extract_data', 'filter_results')",
      "description": "A detailed description of what to do",
      "parameters": {{
        "search_query": "Search query for Amazon (if applicable)",
        "url": "URL to navigate to (if applicable)",
        "filters": "Price, rating, or other filters (if applicable)",
        "data_to_extract": "What information to extract (if applicable)"
      }}
    }}
  ],
  "confidence": "How confident you are in your analysis (0.0 to 1.0)"
}}

Examples:
Command: "Find me a Bluetooth speaker under $50"
Response: {{"intent": "search_products", "actions": [{{"type": "search_amazon", "description": "Search for Bluetooth speakers with price filter", "parameters": {{"search_query": "Bluetooth speaker", "filters": "price under 50"}}}}], "confidence": 0.95}}

Command: "Show me the reviews for this product"
Response: {{"intent": "extract_info", "actions": [{{"type": "extract_data", "description": "Extract product reviews", "parameters": {{"data_to_extract": "product reviews"}}}}], "confidence": 0.85}}

Command: "Compare prices for iPhone 15 Pro"
Response: {{"intent": "compare_products", "actions": [{{"type": "search_amazon", "description": "Search for iPhone 15 Pro variants", "parameters": {{"search_query": "iPhone 15 Pro"}}}}], "confidence": 0.9}}

Command: "Go to the Amazon homepage"
Response: {{"intent": "navigate", "actions": [{{"type": "navigate_to_url", "description": "Navigate to Amazon homepage", "parameters": {{"url": "https://www.amazon.com"}}}}], "confidence": 0.95}}

Just provide the JSON response, nothing else.
"""
        
        try:
            # Call LLM API
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.3,
                "stream": False
            }
            
            response = requests.post(self.api_endpoint, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the response from the API
            api_response = result["choices"][0]["text"].strip()
            
            # Parse the JSON response
            parsed_response = json.loads(api_response)
            
            intent = parsed_response["intent"]
            actions = parsed_response["actions"]
            confidence = parsed_response["confidence"]
            
            self.add_action_log(f"LLM Analysis - Intent: {intent}, Confidence: {confidence}")
            
            return {
                "intent": intent,
                "actions": actions,
                "confidence": confidence
            }
                
        except Exception as e:
            self.add_action_log(f"Error in LLM interpretation: {str(e)}")
            # Fallback to basic parsing
            return self.fallback_parsing(user_input)
    
    def fallback_parsing(self, user_input):
        """Fallback parsing method if LLM fails"""
        self.add_action_log("Using fallback parsing method")
        
        user_input_lower = user_input.lower()
        
        # Basic intent detection
        if "go to" in user_input_lower or "navigate to" in user_input_lower:
            intent = "navigate"
            # Extract URL
            url_match = re.search(r'(?:go to|navigate to)\s+(.+)', user_input_lower)
            if url_match:
                url = url_match.group(1).strip()
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                actions = [
                    {
                        "type": "navigate_to_url",
                        "description": f"Navigate to {url}",
                        "parameters": {"url": url}
                    }
                ]
            else:
                actions = []
        elif "search for" in user_input_lower or "find" in user_input_lower:
            intent = "search_products"
            # Extract query
            query_match = re.search(r'(?:search for|find)\s+(.+)', user_input_lower)
            if query_match:
                query = query_match.group(1).strip()
                actions = [
                    {
                        "type": "search_amazon",
                        "description": f"Search Amazon for {query}",
                        "parameters": {"search_query": query}
                    }
                ]
            else:
                actions = []
        else:
            intent = "general"
            actions = [
                {
                    "type": "search_amazon",
                    "description": "General Amazon search",
                    "parameters": {"search_query": user_input}
                }
            ]
        
        return {
            "intent": intent,
            "actions": actions,
            "confidence": 0.5
        }
    
    def execute_action(self, action):
        """Execute a single action"""
        action_type = action["type"]
        description = action["description"]
        parameters = action["parameters"]
        
        self.add_action_log(f"Executing action: {description}")
        
        if action_type == "search_amazon":
            search_query = parameters.get("search_query", "")
            filters = parameters.get("filters", "")
            
            if search_query:
                # Apply filters to search query if provided
                if filters:
                    search_query = f"{search_query} {filters}"
                
                # Format search URL for Amazon
                encoded_terms = urllib.parse.quote(search_query)
                search_url = f"https://www.amazon.com/s?k={encoded_terms}"
                self.web.load(QUrl(search_url))
                self.status_bar.showMessage(f"Searching for: {search_query}")
                self.address_bar.setText(search_url)
                # Set pending search query for auto-navigation
                self.pending_search_query = search_query
                self.add_action_log(f"Searching Amazon for: {search_query}")
        
        elif action_type == "navigate_to_url":
            url = parameters.get("url", "")
            if url:
                self.web.load(QUrl(url))
                self.status_bar.showMessage(f"Navigating to: {url}")
                self.address_bar.setText(url)
                self.add_action_log(f"Navigating to: {url}")
        
        elif action_type == "extract_data":
            data_to_extract = parameters.get("data_to_extract", "")
            if data_to_extract:
                self.add_action_log(f"Would extract data: {data_to_extract}")
                self.status_bar.showMessage(f"Extracting data: {data_to_extract}")
        
        elif action_type == "filter_results":
            filters = parameters.get("filters", "")
            if filters:
                self.add_action_log(f"Would apply filters: {filters}")
                self.status_bar.showMessage(f"Applying filters: {filters}")
    
    def execute_ai_command(self):
        """Execute the AI command entered by the user"""
        command = self.ai_command.toPlainText().strip()
        if not command:
            self.add_action_log("Please enter a command first.")
            return
        
        self.add_action_log(f"Executing AI command: {command}")
        
        # Use LLM to interpret the command
        interpretation = self.interpret_user_input_with_llm(command)
        
        if interpretation and interpretation["confidence"] > 0.3:
            # Execute the actions
            for action in interpretation["actions"]:
                self.execute_action(action)
        else:
            self.add_action_log("Low confidence in interpretation, please try rephrasing your command.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    browser = LLMAmazonBrowser()
    browser.show()
    sys.exit(app.exec_())
