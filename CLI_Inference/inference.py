import pandas as pd
import joblib
import os
from pathlib import Path

def load_model(model_path: str = "model/addiction_score_pipeline.tuned.joblib"):
    """Load the trained pipeline from disk."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model

class AddictionPredictor:
    def __init__(self):
        # Load the model
        model_path = os.path.join(os.path.dirname(__file__), "../model/addiction_score_pipeline.tuned.joblib")
        self.model = load_model(model_path)
        print("Model loaded successfully!")

    def predict_score(self, user_data):
        df = pd.DataFrame(user_data)
        return self.model.predict(df)

    def categorize(self, score):
        if score <= 3:
            return "Low"
        elif score <= 6:
            return "Moderate"
        else:
            return "High"

    def get_user_input(self):
        """Ask the user for input (simple beginner-friendly way)."""
        user_data = {}
        try:
            user_data['Age'] = int(input("Enter Age (e.g., 20): "))
            user_data['Gender'] = input("Enter Gender (e.g., Male, Female): ")
            user_data['Academic_Level'] = input("Enter Academic Level (e.g., Undergraduate, High School, Postgraduate): ")
            user_data['Avg_Daily_Usage_Hours'] = float(input("Enter Avg. Daily Usage Hours (e.g., 5.5): "))
            user_data['Most_Used_Platform'] = input("Enter Most Used Platform (e.g., Instagram, Facebook, TikTok): ")
            user_data['Affects_Academic_Performance'] = input("Does social media affect academic performance? (Yes/No): ")
            user_data['Sleep_Hours_Per_Night'] = float(input("Enter Sleep Hours Per Night (e.g., 6.0): "))
            user_data['Mental_Health_Score'] = int(input("Enter Mental Health Score (0-10, e.g., 5): "))
            user_data['Relationship_Status'] = input("Enter Relationship Status (e.g., In Relationship, Single, Complicated): ")
            user_data['Conflicts_Over_Social_Media'] = int(input("Enter Conflicts Over Social Media (0-10, e.g., 2): "))
        except ValueError:
            print("Invalid input! Please enter numbers where required.")
            return None
        return [user_data]

# ------------------ Run Example ------------------
if __name__ == "__main__":
    predictor = AddictionPredictor()
    user_input = predictor.get_user_input()
    if user_input:
        score = predictor.predict_score(user_input)[0]
        level = predictor.categorize(score)

        print("\n--- Prediction Results ---")
        print(f"Predicted Score: {score:.2f}")
        print(f"Addiction Level: {level}")
