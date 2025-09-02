import joblib

# Load saved model
model_pipeline = joblib.load("addiction_score_pipeline.tuned.joblib")
print("Loaded trained model")

# Prediction
def predict_addiction_score(new_data):
    new_data_df = pd.DataFrame(new_data)
    predictions = model_pipeline.predict(new_data_df)
    return predictions

def categorize_addiction(score):
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Moderate"
    else:
        return "High"

def get_user_data():
    print("Please enter the following information to predict the social media addiction score:")
    
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
        print("\nInvalid input. Please enter numbers for numeric fields.")
        return None
        
    return [user_data]


# Run
user_input_data = get_user_data()
if user_input_data:
    predicted_score = predict_addiction_score(user_input_data)
    level = categorize_addiction(predicted_score[0])

    print("\n--- Prediction Results ---")
    print(f"Predicted Score: {predicted_score[0]:.2f}")
    print(f"Addiction Level: {level}")
