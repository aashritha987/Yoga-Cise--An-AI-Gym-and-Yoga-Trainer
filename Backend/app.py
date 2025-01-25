from flask import Flask, Response, request, jsonify
from utils import GymExerciseRecognizer,YogaExerciseRecognizer
import cv2
from flask_cors import CORS
import google.generativeai as genai

genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)
CORS(app)

feed = None

@app.route('/video_feed')
def video_feed():
    global feed
    user_choice=request.args.get('id')
    flag=request.args.get('flag')
    if feed is None:
        feed = cv2.VideoCapture(0)
    if flag=='yoga':
        yoga = YogaExerciseRecognizer(user_choice)
        response = Response(yoga.generate_frames(feed), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        response = Response(GymExerciseRecognizer().generate_frames(feed,user_choice), mimetype='multipart/x-mixed-replace; boundary=frame')
    return response

@app.route('/cancel_feed')
def cancel_feed():
    global feed
    if feed is not None:
        feed.release()
        feed = None
        print("Feed released successfully.")
    else:
        print("No feed to release.")
    return "Feed canceled", 200

@app.route('/feedback', methods=['GET'])
def get_feedback():
    recognizer = GymExerciseRecognizer()
    return recognizer.get_feed()

@app.route('/bmi', methods=['GET'])
def bmi_feedback():
    try:
        bmi = float(request.args.get('bmi'))
        if bmi < 18.5:
            category = "underweight"
        elif 18.5 <= bmi < 24.9:
            category = "normal"
        elif 25 <= bmi < 29.9:
            category = "overweight"
        else:
            category = "obese"
        prompt = (
            f"Based on a BMI of {bmi}, which falls into the '{category}' category, "
            "provide 1. food recommendations, 2. exercise recommendations, "
            "3. The daily caloric intake and 4. burn suggestions.The format is important and for every heading 1 line is recomended."
        )
        response = model.generate_content(prompt)
        recommendations = response.text.strip().split('\n')
        sections = response.text.strip().split("##")

        # Create a dictionary to store sections
        response_dict = {}

        for section in sections:
            if section.strip():  # Skip empty sections
                # Separate the title from the content
                title, content = section.split("\n", 1)
                response_dict[title.strip()] = content.strip()

        # Print extracted sections
        food_recommendations = response_dict.get("Food Recommendations")
        exercise_recommendations = response_dict.get("Exercise Recommendations")
        daily_caloric_intake = response_dict.get("Daily Caloric Intake")
        burn_suggestions = response_dict.get("Burn Suggestions")

        return jsonify({
            "bmi_category": category,
            "recommended_foods": food_recommendations,
            "recommended_exercises": exercise_recommendations,
            "daily_caloric_intake": daily_caloric_intake,
            "calories_burned": burn_suggestions
        })

    except ValueError:
        return jsonify({"error": "Invalid BMI value provided."}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)