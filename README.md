💸 AI-Powered Personal Finance Tracker
An intelligent finance tracker that automatically categorizes transactions, detects anomalies, and predicts future spending using machine learning.
Features

Automatic currency detection for multiple currencies (USD, EUR, INR, etc.)
AI-powered transaction categorization using TF-IDF and cosine similarity
Anomaly detection with IsolationForest to identify suspicious transactions
Spending pattern clustering with KMeans to visualize expense trends
Next month spending prediction using LinearRegression
Interactive visualizations with Plotly (pie charts, scatter plots)
User-friendly category management with persistent storage

Requirements

Python 3.8+
Streamlit
Pandas
Plotly
scikit-learn
NumPy

Installation
Follow these steps to set up the project on your local machine.

Clone the repository  
git clone https://github.com/Rishikeshoza11/Ai-Finance-Tracker.git



Navigate to the project directory  
cd Ai-Finance-Tracker



Install the required dependencies  
pip install -r requirements.txt



Run the Streamlit app  
streamlit run app.py

Access the app at http://localhost:8501.


Usage

Upload a CSV file with columns: Date, Details, Amount, Debit/Credit.
Explore the dashboard for spending insights, anomaly detection, and category suggestions.
Use the category editor to manage transaction categories and save changes.

Screenshots
Dashboard Overview
View total spent, received, predicted spending, and a pie chart breakdown by category.
Anomaly Detection and Spending Clusters
See suspicious transactions and spending patterns visualized with KMeans clustering.
Project Structure

app.py: Main Streamlit application code.
requirements.txt: Dependency list.
categories.json: Persistent storage for categories.

Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the project!
License
MIT License - See LICENSE for details.
Contact

GitHub: Rishikeshoza11
Email: [your-email@example.com]
LinkedIn: [your-linkedin-profile]

