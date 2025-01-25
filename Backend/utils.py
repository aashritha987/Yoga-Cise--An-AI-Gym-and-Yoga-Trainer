import mediapipe as mp
import cv2
from flask import jsonify
import numpy as np
import time
import csv

feedback_output = ""
counter_output = 0
state_output = "Down"

def calc_angle(x, y, z):
        x, y, z = np.array(x), np.array(y), np.array(z)
        radians = np.arctan2(z[1] - y[1], z[0] - y[0]) - np.arctan2(x[1] - y[1], x[0] - y[0])
        angle = np.abs(np.degrees(radians))
        if angle > 180.0:
            angle = 360.0 - angle
        return angle

class GymExerciseRecognizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.counter = 0
        self.state = "Down"
        self.feedback = ""
    
    def recognise_squat(self, detection):
        try:
            global feedback_output
            global counter_output
            global state_output

            landmarks = detection.pose_landmarks.landmark

            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            left_knee_angle = calc_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calc_angle(right_hip, right_knee, right_ankle)
            left_hip_angle = calc_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = calc_angle(right_shoulder, right_hip, right_knee)
            back_angle = calc_angle(left_shoulder, left_hip, left_knee)

            if back_angle < 160:
                self.feedback = ' Keep your back straighter! '
            if left_hip_angle < 165 or right_hip_angle < 165:
                self.feedback = ' Drop your hips more! '

            if left_knee_angle > 170 and right_knee_angle > 170:
                self.state = "Up"

            if left_knee_angle < 165 and right_knee_angle < 165:
                self.feedback = ' Almost there... lower until height of hips! '

            if left_knee_angle < 140 and right_knee_angle < 140 and self.state == "Up":
                self.state = "Down"
                self.counter = 1

            if self.state == "Down":
                self.feedback = ' Good rep! '

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in squat detection: {e}")

    def recognise_situp(self, detection):
        try: 
            global feedback_output
            global counter_output
            global state_output

            landmarks = detection.pose_landmarks.landmark
            
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_heel = [landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            angle_knee = calc_angle(left_hip, left_knee, left_heel)
            angle_body = calc_angle(left_shoulder, left_hip, left_knee)
            
            halfway = False
            range_flag = False

            if (angle_body < 80 and angle_body > 50) and self.state == "Down":
                halfway = True
                self.feedback = "Good! You're halfway up. Keep going!"

            if angle_body < 40 and self.state == "Down":
                self.state = "Up"
                range_flag = True
                self.feedback = "Great! You've reached the top."

            if angle_body > 90 and angle_knee < 60:
                self.state = "Down"
                
                if halfway:
                    if range_flag:
                        self.counter += 1
                        self.feedback = "Good repetition! Remember to maintain smooth motion."
                    else:
                        self.feedback = "Incomplete sit-up. Ensure full range of motion next time."
                    range_flag = False
                    halfway = False

            if angle_knee > 70:
                self.feedback = "Your legs are too extended. Keep them tucked closer for better form."

            if angle_body > 100:
                self.feedback = "You're arching your back too much. Try to keep your spine neutral."

            if angle_knee < 50 and angle_body > 120:
                self.feedback = "Too much bending! Keep your body in better alignment."

            if angle_body > 90 and self.state == "Up":
                self.feedback = "You're lowering too slowly. Aim for a controlled, steady pace."

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except: 
            pass

    def recognise_curl(self, detection):
        global feedback_output
        global counter_output
        global state_output
        
        try:
            range_flag=True
            landmarks = detection.pose_landmarks.landmark

            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            
            left_elbow_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calc_angle(right_shoulder, right_elbow, right_wrist)
            
            if left_elbow_angle > 160 and right_elbow_angle > 160:
                if not range_flag:
                    self.feedback = "Incomplete curl. Make sure to curl fully."
                else:
                    self.feedback = ""
                self.state = 'Down'
                feedback_output = self.feedback

            elif (left_elbow_angle > 70 and right_elbow_angle > 70) and self.state == 'Down':
                range_flag = False
                self.feedback = "Not fully curled. Bring your hands closer to your shoulders."
            
            elif (left_elbow_angle < 50 and right_elbow_angle < 50) and self.state == 'Down':
                self.state = 'Up'
                self.feedback = "Good! Full curl achieved."
                range_flag = True
                self.counter += 1
            
            if left_elbow_angle < 160 and right_elbow_angle < 160 and self.state == 'Down':
                self.feedback = "Keep your arms fully extended before curling."

            if left_elbow_angle > 30 and right_elbow_angle < 30:
                self.feedback = "Ensure both arms move evenly. Right arm is lagging behind."

            if left_elbow_angle < 30 and right_elbow_angle > 30:
                self.feedback = "Ensure both arms move evenly. Left arm is lagging behind."

            if left_wrist[1] > left_elbow[1] or right_wrist[1] > right_elbow[1]:
                self.feedback = "Avoid swinging your arms. Control your movement for better form."

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            pass


    def recognise_pushup(self, detection):
        global counter_output
        global state_output
        global feedback_output

        try:
            landmarks = detection.pose_landmarks.landmark

            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            elbow_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
            
            halfway = False
            range_flag = False

            if (elbow_angle < 90 and elbow_angle > 60) and self.state == "Up":
                halfway = True
                self.feedback = "Good! You're halfway down. Keep your form steady."

            if elbow_angle < 45 and self.state == "Up":
                self.state = "Down"
                range_flag = True
                self.feedback = "Well done! Now push up smoothly."

            if elbow_angle > 160:
                self.state = "Up"
                
                if halfway:
                    if range_flag:
                        self.counter += 1
                        self.feedback = "Good repetition! Maintain this form."
                    else:
                        self.feedback = "Incomplete push-up. Try to go lower next time."
                    range_flag = False
                    halfway = False

            if elbow_angle < 30:
                self.feedback = "Elbows too close. Maintain better form by keeping them slightly flared."

            if elbow_angle < 45 and elbow_angle > 30:
                self.feedback = "Try to lower your body further to complete the push-up."

            if elbow_angle > 160 and self.state == "Down":
                self.feedback = "You're pushing too fast. Try a slower, controlled motion for better muscle engagement."

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            pass

    def recognise_lunges(self, detection):
        global counter_output
        global state_output
        global feedback_output

        try:
            landmarks = detection.pose_landmarks.landmark

            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_heel = [landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_heel = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

            left = calc_angle(left_hip, left_knee, left_heel)
            right = calc_angle(right_hip, right_knee, right_heel)

            if left > 150 or right > 150:
                self.state = "Up"
                self.feedback = "You're fully upright. Get ready for the next rep."

            if left < 130 or right < 130:
                self.feedback = "Almost there... lower your knees a bit more."

            if (left < 110 or right < 110) and self.state == "Up":
                self.state = "Down"
                self.counter += 1
                self.feedback = "Good rep! Keep that form steady."

            if left < 90 or right < 90:
                self.feedback = "Knees are too far forward. Keep your front knee aligned with your ankle."

            if abs(left_knee[0] - left_heel[0]) > 0.1 or abs(right_knee[0] - right_heel[0]) > 0.1:
                self.feedback = "Watch your balance! Keep your knee and ankle aligned for stability."

            if self.state == "Down" and (left > 140 or right > 140):
                self.feedback = "Good push-up! Now reset for the next lunge."

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in lunge detection: {e}")
            pass


    def recognise_glutes(self, detection):
        global counter_output
        global state_output
        global feedback_output

        try:
            landmarks = detection.pose_landmarks.landmark
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            
            hip_angle = calc_angle(right_shoulder, right_hip, right_knee)

            if hip_angle >= 165:
                self.state = "Up"
                self.feedback = "Hips fully extended, good job! Keep your core engaged and avoid overarching your lower back."

            if hip_angle < 115 and self.state == "Up":
                self.state = "Down"
                self.counter += 1
                self.feedback = "Great rep! Squeeze your glutes at the top for maximum activation."

            if hip_angle < 90:
                self.feedback = "Lower your hips more for a full range of motion. Keep your knees in line with your hips for proper form."

            if landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x - landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x > 0.1:
                self.feedback = "Watch your knee alignment—make sure they don’t flare outward."

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            pass

    def recognise_pullups(self, detection):
        global counter_output
        global state_output
        global feedback_output

        try:
            landmarks = detection.pose_landmarks.landmark
            
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            elbow_angle_pullups = calc_angle(right_shoulder, right_elbow, right_wrist)

            if elbow_angle_pullups >= 165:
                self.state = "Down"
                self.feedback = "Arms fully extended. Initiate the movement by pulling your shoulder blades down and together before bending your elbows."

            if elbow_angle_pullups <= 20 and self.state == "Down":
                self.state = "Up"
                self.counter += 1
                self.feedback = "Great! Chin over the bar, maintain a straight line from your shoulders to your hips for a solid rep."

            if elbow_angle_pullups > 130 and elbow_angle_pullups < 165:
                self.feedback = "Pull harder to reach full height. Focus on using your back muscles rather than just your arms."

            if (right_wrist[0] - right_elbow[0]) > 0.1:
                self.feedback = " Elbows flaring out too much—keep them closer to your sides for better back engagement."

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            pass

    def recognise_crunches(self, detection):
        global counter_output
        global state_output
        global feedback_output

        try:
            landmarks = detection.pose_landmarks.landmark

            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            shoulder_angle = calc_angle(right_elbow, right_shoulder, right_hip)

            if shoulder_angle >= 165:
                self.state = "Down"
                self.feedback = "Lower back fully on the ground. Keep your core tight before initiating the next rep."

            if shoulder_angle <= 30 and self.state == "Down":
                self.state = "Up"
                self.counter += 1
                self.feedback = "Good crunch! Keep your chin slightly tucked and avoid pulling on your neck."

            if shoulder_angle > 40 and shoulder_angle < 165:
                self.feedback = "Lift your shoulders higher for a full crunch. Engage your core and not your neck."

            if landmarks[self.mp_pose.PoseLandmark.NOSE.value].y - landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y < 0.1:
                self.feedback = " Watch your neck position—avoid curling it forward to prevent strain."

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            pass

    def recognise_side_bend(self, detection):
        global counter_output
        global state_output
        global feedback_output

        try:
            landmarks = detection.pose_landmarks.landmark

            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            hip_angle = calc_angle(left_shoulder, left_hip, left_knee)

            if hip_angle >= 172:
                self.state = "Up"
                self.feedback = "Good posture! Keep your shoulders stacked over your hips for balance."

            if hip_angle <= 163 and self.state == "Up":
                self.state = "Down"
                self.counter += 1
                self.feedback = "Good bend! Feel the stretch on the opposite side and engage your obliques."

            if hip_angle < 170 and self.state == "Up":
                self.feedback = "Bend deeper to get a full range of motion and better activate your obliques."

            if abs(left_shoulder[1] - left_hip[1]) < 0.05:
                self.feedback = " Try to keep your spine straight while bending to avoid unnecessary strain."

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            pass


    def recognise_arm_delt_fly(self, detection):
        global counter_output
        global state_output
        global feedback_output

        try:
            landmarks = detection.pose_landmarks.landmark

            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]

            shoulder_angle = calc_angle(left_elbow, left_shoulder, left_hip)

            if shoulder_angle >= 130:
                self.state = "Up"
                self.feedback = "Good lift! Keep your shoulders down and avoid shrugging to protect your neck."

            if shoulder_angle <= 20 and self.state == "Up":
                self.state = "Down"
                self.counter += 1
                self.feedback = "Great! Lower your arms slowly, controlling the descent for better muscle engagement."

            if shoulder_angle > 70 and shoulder_angle < 130:
                self.feedback = "Keep your arms steady and don’t swing your body. Engage your core to avoid leaning forward."

            if abs(left_shoulder[1] - left_hip[1]) > 0.05:
                self.feedback = "Watch your posture. Ensure you're standing upright with your core engaged."

            if abs(left_elbow[0] - left_shoulder[0]) > 0.05:
                self.feedback = " Avoid locking your elbows—keep a slight bend for proper shoulder engagement."

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            pass

    
    def recognise_deadlift(self, detection):
        try:
            global feedback_output, counter_output, state_output

            landmarks = detection.pose_landmarks.landmark
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angles
            left_knee_angle = calc_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calc_angle(right_hip, right_knee, right_ankle)
            back_angle = calc_angle(left_shoulder, left_hip, left_knee)

            # Feedback logic
            if back_angle > 160:
                self.feedback = ' Keep your back straight! '
            if left_knee_angle > 170 and right_knee_angle > 170:
                self.state = "Up"
            if left_knee_angle < 165 and right_knee_angle < 165:
                self.feedback = ' Almost there... keep lowering! '
            if left_knee_angle < 140 and right_knee_angle < 140 and self.state == "Up":
                self.state = "Down"
                self.counter += 1
            if self.state == "Down":
                self.feedback = ' Good rep! '

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in deadlift detection: {e}")

    def recognise_bench_press(self, detection):
        try:
            global feedback_output, counter_output, state_output

            landmarks = detection.pose_landmarks.landmark
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles
            left_elbow_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

            # Feedback logic
            if left_elbow_angle < 90 or right_elbow_angle < 90:
                self.feedback = ' Lower the bar more! '
            if left_elbow_angle > 160 and right_elbow_angle > 160:
                self.state = "Up"
            if left_elbow_angle < 100 and right_elbow_angle < 100 and self.state == "Up":
                self.state = "Down"
                self.counter += 1
            if self.state == "Down":
                self.feedback = ' Good rep! '

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in bench press detection: {e}")

    def recognise_leg_press(self, detection):
        try:
            global feedback_output, counter_output, state_output

            landmarks = detection.pose_landmarks.landmark
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angles
            left_knee_angle = calc_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calc_angle(right_hip, right_knee, right_ankle)

            # Feedback logic
            if left_knee_angle > 170 and right_knee_angle > 170:
                self.state = "Up"
            if left_knee_angle < 140 and right_knee_angle < 140 and self.state == "Up":
                self.state = "Down"
                self.counter += 1
                self.feedback = ' Good rep! '
            if left_knee_angle < 160 and right_knee_angle < 160:
                self.feedback = ' Lower your knees more! '

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in leg press detection: {e}")

    def recognise_tricep_dips(self, detection):
        try:
            global feedback_output, counter_output, state_output

            landmarks = detection.pose_landmarks.landmark
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles
            left_elbow_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

            # Feedback logic
            if left_elbow_angle < 90 and right_elbow_angle < 90:
                self.state = "Down"
                self.counter += 1
                self.feedback = ' Good rep! '
            if left_elbow_angle > 140 and right_elbow_angle > 140:
                self.state = "Up"
            if left_elbow_angle < 100 and right_elbow_angle < 100:
                self.feedback = ' Lower down more! '

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in tricep dips detection: {e}")

    def recognise_overhead_press(self, detection):
        try:
            global feedback_output, counter_output, state_output

            landmarks = detection.pose_landmarks.landmark
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles
            left_elbow_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

            # Feedback logic
            if left_elbow_angle < 90 and right_elbow_angle < 90:
                self.state = "Up"
                self.counter += 1
                self.feedback = ' Good rep! '
            if left_elbow_angle > 160 and right_elbow_angle > 160:
                self.state = "Down"
            if left_elbow_angle < 100 and right_elbow_angle < 100:
                self.feedback = ' Lower down more! '

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in overhead press detection: {e}")

    def recognise_plank(self, detection):
        try:
            global feedback_output, counter_output, state_output

            landmarks = detection.pose_landmarks.landmark
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Check alignment
            if abs(left_shoulder[1] - right_shoulder[1]) < 0.05 and abs(left_hip[1] - right_hip[1]) < 0.05:
                self.feedback = ' Great plank position! '
                self.state = "Plank"
                self.counter += 1
            else:
                self.feedback = ' Keep your body straight! '

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in plank detection: {e}")

    def recognise_wall_sit(self, detection):
        try:
            global feedback_output, counter_output, state_output

            landmarks = detection.pose_landmarks.landmark
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Check angles
            left_knee_angle = calc_angle(left_knee, left_ankle, left_knee)
            right_knee_angle = calc_angle(right_knee, right_ankle, right_knee)

            if left_knee_angle < 90 and right_knee_angle < 90:
                self.feedback = ' Great wall sit position! '
                self.state = "Sitting"
                self.counter += 1
            else:
                self.feedback = ' Lower down more! '

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in wall sit detection: {e}")

    def recognise_calf_raise(self, detection):
        try:
            global feedback_output, counter_output, state_output

            landmarks = detection.pose_landmarks.landmark
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            if abs(left_ankle[1] - left_knee[1]) < 0.05 and abs(right_ankle[1] - right_knee[1]) < 0.05:
                self.feedback = ' Good calf raise position! '
                self.state = "Raising"
                self.counter += 1
            else:
                self.feedback = ' Raise higher! '

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in calf raise detection: {e}")

    def recognise_high_knees(self, detection):
        try:
            global feedback_output, counter_output, state_output

            landmarks = detection.pose_landmarks.landmark
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # Check height of knees
            if left_knee[1] > 0.5 and right_knee[1] > 0.5:
                self.feedback = ' Good high knees! '
                self.state = "High Knees"
                self.counter += 1
            else:
                self.feedback = ' Raise your knees higher! '

            state_output = self.state
            feedback_output = self.feedback
            counter_output = self.counter

        except Exception as e:
            print(f"Error in high knees detection: {e}")


    def generate_frames(self,feed,user_choice):

        global feedback_output
        global counter_output
        global state_output
        frame_count = 0
        display_width = 960
        display_height = 540
        feed.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        feed.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        width = int(feed.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(feed.get(cv2.CAP_PROP_FRAME_HEIGHT))
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while feed.isOpened():
                ret, frame = feed.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame_count += 1
                image = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                detection = pose.process(image)
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                if detection.pose_landmarks:
                    visibility_threshold = 0.5
                    meaningful_landmarks = [lm for lm in detection.pose_landmarks.landmark if lm.visibility > visibility_threshold]
                    num_meaningful_landmarks = len(meaningful_landmarks)
                    if num_meaningful_landmarks>18:
                        image.flags.writeable = True
                        landmark_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=5)
                        connection_spec = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
                        self.mp_drawing.draw_landmarks(image, detection.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=landmark_spec,connection_drawing_spec=connection_spec)
                    
                        exercise_recognition_map = {
                            "0001": self.recognise_squat,
                            "0002": self.recognise_lunges,
                            "0003": self.recognise_crunches,
                            "0004": self.recognise_situp,
                            "0005": self.recognise_side_bend,
                            "0006": self.recognise_curl,
                            "0007": self.recognise_pushup,
                            "0008": self.recognise_pullups,
                            "0009": self.recognise_arm_delt_fly,
                            "00010": self.recognise_glutes,
                            "00011": self.recognise_deadlift,
                            "00012": self.recognise_bench_press,
                            "00013": self.recognise_leg_press,
                            "00014": self.recognise_tricep_dips,
                            "00015": self.recognise_overhead_press,
                            "00016": self.recognise_plank,
                            "00017": self.recognise_wall_sit,
                            "00018": self.recognise_calf_raise,
                            "00019": self.recognise_high_knees
                        }
                        exercise_recognition_map.get(user_choice, lambda x: None)(detection)
                        cv2.rectangle(image, (0,0), (width, int(height*0.1)), (245,117,16), -1)
                        cv2.putText(image, "REPS:", (int(width*0.01), int(height*0.025)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(image, "STATE:", (int(width*0.1), int(height*0.025)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(image, "FEEDBACK:", (int(width*0.2), int(height*0.025)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(image, str(counter_output), (int(width*0.01), int(height*0.08)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                        cv2.putText(image, state_output, (int(width*0.1), int(height*0.08)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                        cv2.putText(image, feedback_output, (int(width*0.2), int(height*0.08)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def get_feed(self):
        global feedback_output
        global counter_output
        global state_output
        return jsonify({
            'feedback': feedback_output,
            'counter': counter_output,
            'state': state_output
        })
    

class YogaExerciseRecognizer:
    def __init__(self,user_choice):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.counter = 0
        self.state = "Down"
        self.feedback = ""
        self.accurate_angle_list=[]
        exercise_recognition_map = {
                            "1": "Baddha Konasana",
                            "2": "Ardha Chandrasana",
                            "3": "Adho Mukha Svanasana",
                            "4": "Utkata Konasana",
                            "5": "Natarajasana",
                            "6": "Kumbhakasana",
                            "7": "Vrikshasana",
                            "8":"Utthita Trikonasana",
                            "9":"Virabhadrasana I",
                            "10":"Virabhadrasana II",
                            "11": "Bhujangasana",
                            "12": "Dhanurasana",
                            "13": "Setu Bandhasana",
                            "14": "Pavanamuktasana",
                            "15": "Balasana",
                            "16": "Paschimottanasana",
                            "17": "Savasana",
                            "18": "Trikonasana",
                            "19": "Salamba Sarvangasana",
                            "20":"Matsyasana"
                        }
        with open('angle_teacher_yoga.csv', 'r') as inputCSV:
            reader = csv.reader(inputCSV)
            next(reader)
            for row in reader:
                if row[12]==exercise_recognition_map[user_choice]:
                    self.accurate_angle_list.append([
                        float(row[0]), float(row[1]), float(row[2]), 
                        float(row[3]), float(row[4]), float(row[5]), 
                        float(row[6]), float(row[7]), float(row[8]), 
                        float(row[9]), float(row[10]), float(row[11]), row[12]
                    ])
                    self.accurate_angle_list = self.accurate_angle_list[0]
                    break
            if self.accurate_angle_list is []:
                self.accurate_angle_list = self.accurate_angle_list[0]
    
    def calculate_angle(self,landmark1, landmark2, landmark3,select=''):
        if select == '1':
            x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
            x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
            x3, y3, _ = landmark3.x, landmark3.y, landmark3.z
            angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
        else:
            radians = np.arctan2(landmark3[1] - landmark2[1], landmark3[0] - landmark2[0]) - np.arctan2(landmark1[1] - landmark2[1], landmark1[0] - landmark2[0])
            angle = np.abs(np.degrees(radians))
        angle_calc = angle + 360 if angle < 0 else angle
        return angle_calc

    def generate_frames(self,cap):
        angle_name_list = ["L-wrist","R-wrist","L-elbow", "R-elbow","L-shoulder", "R-shoulder", "L-knee", "R-knee","L-ankle","R-ankle","L-hip", "R-hip"]
        angle_coordinates = [[13, 15, 19], [14, 16, 18], [11, 13, 15], [12, 14, 16], [13, 11, 23], [14, 12, 24], [23, 25, 27], [24, 26, 28],[23,27,31],[24,28,32],[24,23,25],[23,24,26]]
        correction_value = 30
        fps_time = 0        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                success, image = cap.read()
                if not success:
                    break
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resize_rgb = cv2.resize(image_rgb, (0, 0), None, .80, .80)
                results = pose.process(resize_rgb)
                angles = []
                if results.pose_landmarks is not None:
                    landmarks = results.pose_landmarks.landmark
                    left_wrist_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX.value],'1')
                    angles.append(left_wrist_angle)
                    right_wrist_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX.value],'1')
                    angles.append(right_wrist_angle)
                    left_elbow_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value],'1')
                    angles.append(left_elbow_angle)
                    right_elbow_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value],'1')
                    angles.append(right_elbow_angle)
                    left_shoulder_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],'1')
                    angles.append(left_shoulder_angle)
                    right_shoulder_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],'1')
                    angles.append(right_shoulder_angle)
                    left_knee_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],'1')
                    angles.append(left_knee_angle)
                    right_knee_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value],'1')
                    angles.append(right_knee_angle)
                    left_ankle_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value],'1')
                    angles.append(left_ankle_angle)
                    right_ankle_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value],'1')
                    angles.append(right_ankle_angle)
                    left_hip_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],'1')
                    angles.append(left_hip_angle)
                    right_hip_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],'1')
                    angles.append(right_hip_angle)
                    correct_angle_count = 0
                    for itr in range(12):
                        point_a = (int(landmarks[angle_coordinates[itr][0]].x * image.shape[1]),
                                int(landmarks[angle_coordinates[itr][0]].y * image.shape[0]))
                        point_b = (int(landmarks[angle_coordinates[itr][1]].x * image.shape[1]),
                                int(landmarks[angle_coordinates[itr][1]].y * image.shape[0]))
                        point_c = (int(landmarks[angle_coordinates[itr][2]].x * image.shape[1]),
                                int(landmarks[angle_coordinates[itr][2]].y * image.shape[0]))
                        angle_obtained = self.calculate_angle(point_a, point_b, point_c,'0')
                        if angle_obtained < self.accurate_angle_list[itr] - correction_value:
                            status = "more"
                        elif self.accurate_angle_list[itr] + correction_value < angle_obtained:
                            status = "less"
                        else:
                            status = "OK"
                            correct_angle_count += 1
                        status_position = (point_b[0] - int(image.shape[1] * 0.03), point_b[1] + int(image.shape[0] * 0.03))
                        cv2.putText(image, f"{status}", status_position, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                        cv2.putText(image, f"{angle_name_list[itr]}", (point_b[0] - 50, point_b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                    mp_drawing = mp.solutions.drawing_utils
                    landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=5)
                    connection_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=landmark_spec,connection_drawing_spec=connection_spec)
                    posture = "CORRECT" if correct_angle_count > 9 else "WRONG"
                    posture_color = (0, 255, 0) if posture == "CORRECT" else (0, 0, 255)
                    posture_position = (10, 30)
                    cv2.putText(image, f"Yoga movements: {posture}", posture_position, cv2.FONT_HERSHEY_PLAIN, 1.5, posture_color, 2)
                    fps_text = f"FPS: {1.0 / (time.time() - fps_time):.3f}"
                    fps_position = (10, 60)
                    cv2.putText(image, fps_text, fps_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    fps_time = time.time()
                ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')