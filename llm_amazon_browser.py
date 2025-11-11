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
from bs4 import BeautifulSoup

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
        
        # self.home_btn = QPushButton("Amazon Home")
        # self.home_btn.clicked.connect(self.go_home)
        
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
        self.actions_log.setMaximumHeight(800)
        
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
        self.pending_price_constraint = None
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
        #nav_layout.addWidget(self.home_btn)
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
            if self.pending_price_constraint:
                self.add_action_log(f"Price constraint: {self.pending_price_constraint}")
            
            # Use a QTimer to delay the execution of the JavaScript code
            QTimer.singleShot(1000, self._extract_amazon_search_results)
            self.search_results_page = False
    
    def _extract_amazon_search_results(self):
        """Extract Amazon search results after a delay"""
        # Use JavaScript to find all search results and their titles
        js_code = """
        // Get all search results with their titles and content
        var searchResults = [];
        var resultElements = document.querySelectorAll('[data-component-type="s-search-result"]');
        
        if (resultElements.length === 0) {
            // Try alternative selectors
            resultElements = document.querySelectorAll('.s-result-item');
        }
        
        if (resultElements.length === 0) {
            // Try another alternative selector
            resultElements = document.querySelectorAll('.s-search-results .sg-col');
        }
        
        for (var i = 0; i < resultElements.length; i++) {
            var element = resultElements[i];
            
            // Try multiple selectors for the title
            var titleElement = element.querySelector('h2 a span');
            if (!titleElement) {
                titleElement = element.querySelector('h2 span');
            }
            if (!titleElement) {
                titleElement = element.querySelector('.a-text-normal span');
            }
            if (!titleElement) {
                titleElement = element.querySelector('h2 a');
            }
            
            // Get all text content within the result element
            var allText = element.textContent.trim();
            
            // Try to find and append price information
            var priceElement = element.querySelector('.a-price .a-offscreen');
            if (!priceElement) {
                priceElement = element.querySelector('.a-price-whole');
            }
            if (!priceElement) {
                priceElement = element.querySelector('.a-color-price');
            }
            if (!priceElement) {
                priceElement = element.querySelector('.a-price');
            }
            if (priceElement) {
                allText += "\\nPrice: " + priceElement.textContent.trim();
            }
            
            if (titleElement) {
                var title = titleElement.textContent.trim();
                // Only add results with non-empty titles
                if (title) {
                    searchResults.push({
                        title: title,
                        allText: allText
                    });
                }
            }
        }
        
        // Return the search results to Python for LLM processing
        searchResults.map(result => ({
            title: result.title,
            allText: result.allText
        }));
        """
        
        self.web.page().runJavaScript(js_code, self.process_search_results_with_llm)
    
    def process_search_results_with_llm(self, search_results):
        """Process search results using LLM to find the best match"""
        if not search_results:
            self.add_action_log("No search results found")
            self.pending_search_query = None
            self.pending_price_constraint = None
            return
            
        self.add_action_log(f"Processing {len(search_results)} search results with LLM")
        
        # First, use LLM to extract prices and filter based on price constraints
        if self.pending_price_constraint:
            # Create a prompt for the LLM to filter results based on price constraints
            results_with_text = "\n".join([f"{i+1}. {result['title']}\n   Text: {result['allText']}" for i, result in enumerate(search_results)])
            print('----results_with_text----', results_with_text)
            
            price_filter_prompt = f"""
You are an AI assistant helping to filter Amazon search results based on price constraints.
The user is looking for: "{self.pending_search_query}"
Price constraint: "{self.pending_price_constraint}"

Here are the search results with their full text content:
{results_with_text}

Please identify which results meet the price constraint. Extract the price from each result's text and check if it meets the constraint.

Respond with ONLY a JSON array containing the numbers of results that meet the price constraint (most relevant first).
For example, if results 2 and 1 meet the constraint, respond:
[2, 1]

If no results meet the constraint, respond with an empty array:
[]

Just provide the JSON array, nothing else.
"""
            
            try:
                # Call LLM API for price filtering
                payload = {
                    "model": self.model,
                    "prompt": price_filter_prompt,
                    "max_tokens": 500,
                    "temperature": 0.1,
                    "stream": False
                }
                
                response = requests.post(self.api_endpoint, headers=self.headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # Extract the response from the API
                api_response = result["choices"][0]["text"].strip()
                
                # Parse the JSON response
                filtered_indices = json.loads(api_response)
                
                # Filter the search results based on LLM's response
                if filtered_indices:
                    search_results = [search_results[i-1] for i in filtered_indices]
                    self.add_action_log(f"Filtered to {len(search_results)} results based on price constraint")
                else:
                    self.add_action_log("No results meet the price constraint")
                    # Clean up and return
                    self.pending_search_query = None
                    self.pending_price_constraint = None
                    return
                    
            except Exception as e:
                self.add_action_log(f"Error in LLM price filtering: {str(e)}")
                # Continue with all results if price filtering fails
        
        # Create a prompt for the LLM to rank the search results
        results_text = "\n".join([f"{i+1}. {result['title']}" for i, result in enumerate(search_results)])
        
        prompt = f"""
You are an AI assistant helping to find the most relevant product on Amazon. 
The user is looking for: "{self.pending_search_query}"

Here are the search results:
{results_text}

Please rank these results from most relevant (1) to least relevant, considering:
1. How well the title matches the user's query
2. Whether the product is the actual item or an accessory/case/etc.
3. Relevance to the user's intent

Respond with ONLY a JSON array containing the numbers in ranked order (most relevant first).
For example, if there are 3 results and result 2 is most relevant, then 1, then 3, respond:
[2, 1, 3]

Just provide the JSON array, nothing else.
"""
        
        try:
            # Call LLM API
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 32768,
                "temperature": 0.1,
                "stream": False
            }
            
            response = requests.post(self.api_endpoint, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the response from the API
            api_response = result["choices"][0]["text"].strip()
            
            # Parse the JSON response
            ranked_indices = json.loads(api_response)
            
            if ranked_indices and len(ranked_indices) > 0:
                # Validate the ranked indices
                valid_indices = [idx for idx in ranked_indices if 1 <= idx <= len(search_results)]
                if not valid_indices:
                    self.add_action_log("LLM returned invalid indices")
                    # Fallback to clicking the first result
                    self.fallback_click_first_result()
                    return
                
                # Get the index of the best match (first in ranked list)
                best_match_index = valid_indices[0] - 1  # Convert from 1-based to 0-based indexing
                
                if 0 <= best_match_index < len(search_results):
                    best_match = search_results[best_match_index]
                    print('The best match is:', best_match)
                    
                    # Check if the title is empty
                    if not best_match.get('title', '').strip():
                        self.add_action_log("LLM selected best match with empty title")
                        # Fallback to clicking the first result
                        self.fallback_click_first_result()
                        return
                    
                    self.add_action_log(f"LLM selected best match: {best_match['title']}")
                    
                    # Now we need to click on this item
                    # We'll use JavaScript to find and click the element with this title
                    js_click_code = f"""
                    // Find and click element with matching title
                    var resultElements = document.querySelectorAll('[data-component-type="s-search-result"] h2 span, .s-result-item h2 span, [data-component-type="s-search-result"] .a-text-normal span, .s-result-item .a-text-normal span');
                    var targetTitle = "{best_match['title'].replace('"', '\\"')}";

                    for (var i = 0; i < resultElements.length; i++) {{
                        var spanElement = resultElements[i];
                        var title = spanElement.textContent.trim();
                        
                        if (title === targetTitle) {{
                            // Find the parent <a> tag to click
                            var parentLink = spanElement.closest('a');
                            if (parentLink) {{
                                parentLink.click();
                                
                            }}
                        }}
                    }}
                    
                    """

                    self.web.page().runJavaScript(js_click_code, self.js_callback)
                else:
                    self.add_action_log("LLM returned invalid index")
                    # Fallback to clicking the first result
                    self.fallback_click_first_result()
            else:
                self.add_action_log("LLM returned empty ranking")
                # Fallback to clicking the first result
                self.fallback_click_first_result()
                
        except Exception as e:
            self.add_action_log(f"Error in LLM processing: {str(e)}")
            # Fallback to clicking the first result
            self.fallback_click_first_result()
        
        # Clean up
        self.pending_search_query = None
        self.pending_price_constraint = None
    
    def fallback_click_first_result(self):
        """Fallback method to click the first search result"""
        self.add_action_log("Falling back to clicking first result")
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

    def js_callback(self, result):
        """Callback for JavaScript execution"""
        if result:
            self.add_action_log(f"JavaScript result: {result}")

    def search_google_and_analyze(self, search_query):
        """Search Google for the given query and analyze the first 5 results"""
        self.add_action_log(f"Searching Google for: {search_query}")
        self.status_bar.showMessage(f"Searching Google for: {search_query}")
        
        # Encode the search query
        encoded_query = urllib.parse.quote(search_query)
        url = f"https://www.google.com/search?q={encoded_query}&num=5"
        
        # Navigate to the Google search results page
        self.web.load(QUrl(url))
        
        # Store the search query for use in the callback
        self.current_google_search_query = search_query
        
        # Connect to the loadFinished signal to process the results when the page loads
        self.web.loadFinished.connect(self.process_google_search_results)
    
    def process_google_search_results(self, success):
        """Process Google search results after the page has loaded"""
        # Disconnect the signal to prevent it from being called again
        self.web.loadFinished.disconnect(self.process_google_search_results)
        
        if not success:
            self.add_action_log("Failed to load Google search results")
            self.status_bar.showMessage("Failed to load Google search results")
            return
        
        # Use a QTimer to delay the execution of the JavaScript code
        QTimer.singleShot(1000, self._extract_google_search_results)
    
    def _extract_google_search_results(self):
        """Extract Google search results after a delay"""
        # Use JavaScript to extract search results
        js_code = """
        // Extract search results
        var results = [];
        var resultElements = document.querySelectorAll('div.g');
        
        // If the above selector doesn't work, try alternative selectors
        if (resultElements.length === 0) {
            resultElements = document.querySelectorAll('.g');
        }
        
        for (var i = 0; i < Math.min(resultElements.length, 5); i++) {
            var element = resultElements[i];
            
            // Extract title
            var titleElement = element.querySelector('h3');
            if (!titleElement) {
                titleElement = element.querySelector('h3 a');
            }
            var title = titleElement ? titleElement.textContent.trim() : 'No title';
            
            // Extract URL
            var linkElement = element.querySelector('a');
            if (!linkElement) {
                linkElement = element.querySelector('h3 a');
            }
            var url = linkElement ? linkElement.href : 'No URL';
            
            // Extract snippet
            var snippetElement = element.querySelector('.s3v9rd');  // Common class for snippets
            if (!snippetElement) {
                snippetElement = element.querySelector('.st');  // Another common class for snippets
            }
            if (!snippetElement) {
                snippetElement = element.querySelector('span');  // Fallback to any span
            }
            var snippet = snippetElement ? snippetElement.textContent.trim() : 'No snippet';
            
            // Only add results with non-empty titles
            if (title && title !== 'No title') {
                results.push({
                    title: title,
                    url: url,
                    snippet: snippet
                });
            }
        }
        
        // Return results to Python
        results;
        """
        
        # Run the JavaScript code and process the results
        self.web.page().runJavaScript(js_code, self.handle_google_results)
    
    def handle_google_results(self, results):
        """Handle the Google search results and summarize them"""
        if not results:
            self.add_action_log("No Google search results found")
            self.status_bar.showMessage("No Google search results found")
            return
        
        # Summarize the results
        self.summarize_google_results(self.current_google_search_query, results)

    def summarize_google_results(self, search_query, search_results):
        """Summarize Google search results using LLM"""
        self.add_action_log(f"Summarizing {len(search_results)} Google search results")
        
        # Format the search results for the LLM
        results_text = "\n".join([
            f"{i+1}. Title: {result['title']}\n   Snippet: {result['snippet']}\n   URL: {result['url']}"
            for i, result in enumerate(search_results)
        ])
        
        # Create a prompt for the LLM to summarize the results
        prompt = f"""
You are an AI assistant that summarizes Google search results. Provide a concise summary of the following search results for the query: "{search_query}"

Search Results:
{results_text}

Please provide a clear, concise summary that answers the user's query based on the search results. 
Organize the information in a structured way, highlighting the most relevant points.
If there are conflicting viewpoints in the results, mention them.
If the results don't directly answer the query, provide the most relevant information found.

Format your response as a clear summary without any additional formatting or markdown.
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
            summary = result["choices"][0]["text"].strip()
            
            # Display the summary in the actions log
            self.add_action_log(f"Google Search Summary for '{search_query}':")
            self.add_action_log(summary)
            self.status_bar.showMessage("Google search summary completed")
            
            # Create an HTML page with the summary and display it in the web view
            html_content = f"""
            <html>
            <head>
                <title>Google Search Results Summary</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #4285f4; }}
                    .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; }}
                    .results {{ margin-top: 20px; }}
                    .result {{ margin-bottom: 15px; padding: 10px; border-left: 3px solid #4285f4; }}
                    .title {{ font-weight: bold; }}
                    .snippet {{ margin: 5px 0; }}
                    .url {{ color: #0d6efd; }}
                </style>
            </head>
            <body>
                <h1>Google Search Results Summary</h1>
                <h2>Query: {search_query}</h2>
                <div class="summary">
                    <h3>Summary:</h3>
                    <p>{summary.replace(chr(10), '<br>')}</p>
                </div>
                <div class="results">
                    <h3>Top Results:</h3>
                    {''.join([f'''
                    <div class="result">
                        <div class="title">{result['title']}</div>
                        <div class="snippet">{result['snippet']}</div>
                        <div class="url">{result['url']}</div>
                    </div>
                    ''' for result in search_results])}
                </div>
            </body>
            </html>
            """
            
            # Load the HTML content in the web view
            self.web.setHtml(html_content)
            self.address_bar.setText(f"Google Search: {search_query}")
            
        except Exception as e:
            self.add_action_log(f"Error in summarizing results: {str(e)}")
            self.status_bar.showMessage(f"Error in summarizing results: {str(e)}")

    def interpret_user_input_with_llm(self, user_input):
        """Use LLM to interpret user input and determine actions"""
        self.add_action_log(f"Interpreting user input with LLM: {user_input}")
        
        # Create a prompt for the LLM to determine the intent and actions
        prompt = f"""
You are an AI assistant for an Amazon browser. Analyze the following user command and determine what actions the browser should take.

User Command: "{user_input}"

First, determine if this query is shopping-related or not. Shopping-related queries typically involve:
- Searching for products to buy
- Comparing products
- Looking for product information, reviews, prices
- Finding deals or discounts
- Product specifications

Non-shopping queries might involve:
- General information seeking
- News
- Recipes
- How-to questions
- Entertainment
- Travel information
- Weather
- Definitions

Please provide your response in the following JSON format:
{{
  "is_shopping_related": "Boolean indicating if the query is shopping-related (true/false)",
  "intent": "The user's intent (e.g., 'search_products', 'filter_results', 'extract_info', 'compare_products', 'navigate', 'general_search')",
  "actions": [
    {{
      "type": "The type of action (e.g., 'search_amazon', 'navigate_to_url', 'extract_data', 'filter_results', 'search_google')",
      "description": "A detailed description of what to do",
      "parameters": {{
        "search_query": "Main product keywords for Amazon search (e.g., 'iPhone 16 Pro Max')",
        "url": "URL to navigate to (if applicable)",
        "filters": "Price, rating, or other filters (if applicable)",
        "price_constraint": "Price constraint if mentioned (e.g., 'under $1200', 'over $50', 'between $100 and $200')",
        "data_to_extract": "What information to extract (if applicable)"
      }}
    }}
  ],
  "confidence": "How confident you are in your analysis (0.0 to 1.0)"
}}

Examples:
Command: "Find me a Bluetooth speaker under $50"
Response: {{"is_shopping_related": true, "intent": "search_products", "actions": [{{"type": "search_amazon", "description": "Search for Bluetooth speakers with price filter", "parameters": {{"search_query": "Bluetooth speaker", "price_constraint": "under $50"}}}}], "confidence": 0.95}}

Command: "Looking for an iPhone 16 Pro Max under $1200"
Response: {{"is_shopping_related": true, "intent": "search_products", "actions": [{{"type": "search_amazon", "description": "Search for iPhone 16 Pro Max with price filter", "parameters": {{"search_query": "iPhone 16 Pro Max", "price_constraint": "under $1200"}}}}], "confidence": 0.95}}

Command: "Show me the reviews for this product"
Response: {{"is_shopping_related": true, "intent": "extract_info", "actions": [{{"type": "extract_data", "description": "Extract product reviews", "parameters": {{"data_to_extract": "product reviews"}}}}], "confidence": 0.85}}

Command: "Compare prices for iPhone 15 Pro"
Response: {{"is_shopping_related": true, "intent": "compare_products", "actions": [{{"type": "search_amazon", "description": "Search for iPhone 15 Pro variants", "parameters": {{"search_query": "iPhone 15 Pro"}}}}], "confidence": 0.9}}

Command: "Go to the Amazon homepage"
Response: {{"is_shopping_related": true, "intent": "navigate", "actions": [{{"type": "navigate_to_url", "description": "Navigate to Amazon homepage", "parameters": {{"url": "https://www.amazon.com"}}}}], "confidence": 0.95}}

Command: "What's the weather like today?"
Response: {{"is_shopping_related": false, "intent": "general_search", "actions": [{{"type": "search_google", "description": "Search Google for weather information", "parameters": {{"search_query": "weather today"}}}}], "confidence": 0.95}}

Command: "How to make pasta?"
Response: {{"is_shopping_related": false, "intent": "general_search", "actions": [{{"type": "search_google", "description": "Search Google for pasta recipe", "parameters": {{"search_query": "how to make pasta"}}}}], "confidence": 0.95}}

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
            
            is_shopping_related = parsed_response.get("is_shopping_related", True)
            intent = parsed_response["intent"]
            actions = parsed_response["actions"]
            confidence = parsed_response["confidence"]
            
            self.add_action_log(f"LLM Analysis - Shopping Related: {is_shopping_related}, Intent: {intent}, Confidence: {confidence}")
            
            return {
                "is_shopping_related": is_shopping_related,
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
            price_constraint = parameters.get("price_constraint", "")
            
            if search_query:
                # For now, we only search with the main product keywords
                # Price filtering will be handled in the auto-navigation step
                encoded_terms = urllib.parse.quote(search_query)
                search_url = f"https://www.amazon.com/s?k={encoded_terms}"
                self.web.load(QUrl(search_url))
                self.status_bar.showMessage(f"Searching for: {search_query}")
                self.address_bar.setText(search_url)
                # Set pending search query for auto-navigation
                self.pending_search_query = search_query
                # Store price constraint for auto-navigation
                self.pending_price_constraint = price_constraint
                self.add_action_log(f"Searching Amazon for: {search_query}")
                if price_constraint:
                    self.add_action_log(f"Price constraint: {price_constraint}")
        
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
        
        elif action_type == "search_google":
            search_query = parameters.get("search_query", "")
            if search_query:
                self.search_google_and_analyze(search_query)
    
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
            # Check if the query is shopping-related
            is_shopping_related = interpretation.get("is_shopping_related", True)
            
            if not is_shopping_related:
                # For non-shopping queries, we'll search Google directly
                self.add_action_log("Non-shopping query detected, searching Google...")
                # Find the search_google action if it exists
                google_action = None
                for action in interpretation["actions"]:
                    if action["type"] == "search_google":
                        google_action = action
                        break
                
                if google_action:
                    self.execute_action(google_action)
                else:
                    # Fallback: create a search_google action
                    fallback_action = {
                        "type": "search_google",
                        "description": f"Search Google for {command}",
                        "parameters": {"search_query": command}
                    }
                    self.execute_action(fallback_action)
            else:
                # For shopping-related queries, execute the actions as before
                for action in interpretation["actions"]:
                    self.execute_action(action)
        else:
            self.add_action_log("Low confidence in interpretation, please try rephrasing your command.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    browser = LLMAmazonBrowser()
    browser.show()
    sys.exit(app.exec_())
