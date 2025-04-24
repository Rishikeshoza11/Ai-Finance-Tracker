# 💸 AI-Powered Personal Finance Tracker

An intelligent finance tracker that automatically categorizes transactions, detects anomalies, and predicts future spending using machine learning.

## Features
- Automatic currency detection for multiple currencies (USD, EUR, INR, etc.)
- AI-powered transaction categorization using TF-IDF and cosine similarity
- Anomaly detection with IsolationForest to identify suspicious transactions
- Spending pattern clustering with KMeans to visualize expense trends
- Next month spending prediction using LinearRegression
- Interactive visualizations with Plotly (pie charts, scatter plots)
- User-friendly category management with persistent storage

## Requirements
- Python 3.8+
- Streamlit
- Pandas
- Plotly
- scikit-learn
- NumPy

## Installation

Follow these steps to set up the project on your local machine.

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Rishikeshoza11/Ai-Finance-Tracker.git
   ```

2. **Navigate to the project directory**  
   ```bash
   cd Ai-Finance-Tracker
   ```

3. **Install the required dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**  
   ```bash
   streamlit run app.py
   ```
   Access the app at `http://localhost:8501`.

## Usage
- Upload a CSV file with columns: `Date`, `Details`, `Amount`, `Debit/Credit`.
- Explore the dashboard for spending insights, anomaly detection, and category suggestions.
- Use the category editor to manage transaction categories and save changes.

## Screenshots

### Dashboard Overview
![Dashboard Overview](https://github.com/Rishikeshoza11/Ai-Finance-Tracker/blob/ca7215fec395d5ca00136fa6a687bad374959bbc/Screenshot1.png) 
![Expense Breakdown](https://github.com/Rishikeshoza11/Ai-Finance-Tracker/blob/7346001e4818f9713137f4b87937536a81e8d593/Screenshot2.png) 
![Spending Pattern Clustering](https://github.com/Rishikeshoza11/Ai-Finance-Tracker/blob/4d2d5badc16b79312eacc692689a065d365bf320/screenshot3.png) 


*View total spent, received, predicted spending, and a pie chart breakdown by category.*

### Anomaly Detection and Spending Clusters
![Anomaly Detection and Clusters](https://github.com/Rishikeshoza11/Ai-Finance-Tracker/blob/85eb066ac09c0e0b221e528503621676ae05fe98/screenshot4.png)  
*See suspicious transactions and spending patterns visualized with KMeans clustering.*
![Manage Categories](https://github.com/Rishikeshoza11/Ai-Finance-Tracker/blob/543a0b461de97cd3bc17431f75e1cc1797ea20a5/screenshot5.png)
*Manage The categories

## Project Structure
- `app.py`: Main Streamlit application code.
- `requirements.txt`: Dependency list.
- `categories.json`: Persistent storage for categories.

## Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the project!

## License
MIT License - See [LICENSE](LICENSE) for details.

## Contact
- GitHub: [Rishikeshoza11](https://github.com/Rishikeshoza11)
- Email: rishikeshoza77@gmail.com
