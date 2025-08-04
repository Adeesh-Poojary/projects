import pandas as pd
import re
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Downloading dataset...")
url = "https://raw.githubusercontent.com/sid321axn/malicious-urls-dataset/main/malicious_phish.csv"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# Show sample
print("Sample data:")
print(data.head())

# Preprocess dataset
data = data[['url', 'type']]
data['label'] = data['type'].apply(lambda x: 1 if x in ['phishing', 'malicious'] else 0)

# Feature extraction function
def extract_features(url):
    return {
        'url_length': len(url),
        'has_ip': int(bool(re.search(r'\d{1,3}(\.\d{1,3}){3}', url))),
        'count_dots': url.count('.'),
        'count_hyphens': url.count('-'),
        'has_https': int(url.startswith('https')),
        'has_at': int('@' in url),
    }

# Extract features from all URLs
features = data['url'].apply(extract_features)
features_df = pd.DataFrame(features.tolist())
labels = data['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nModel performance:")
print(classification_report(y_test, y_pred))

# Predict on custom input
def predict_url(url):
    features = extract_features(url)
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    print(f"\nPrediction for '{url}':", "Malicious" if pred == 1 else "Safe")

# Try a custom URL
predict_url("http://verify-paypal-login.com")
predict_url("https://www.wikipedia.org")