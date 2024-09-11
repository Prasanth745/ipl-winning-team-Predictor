

from flask import Flask, render_template, request
import pickle
import pandas as pd
import base64

app = Flask(__name__)

# Load the machine learning model
pipe = pickle.load(open('pipe.pkl', 'rb'))

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# List of teams and cities
cities = ['Mumbai', 'Delhi', 'Chennai', 'Abu Dhabi', 'Visakhapatnam', 'Hyderabad', 'Chandigarh',
          'Ahmedabad', 'Bangalore', 'Jaipur', 'Kolkata', 'Port Elizabeth', 'Cuttack', 'Navi Mumbai',
          'Centurion', 'Bengaluru', 'Pune', 'Johannesburg', 'Dubai', 'Cape Town', 'Lucknow', 'Durban',
          'Dharamsala', 'Indore', 'East London', 'Sharjah', 'Guwahati', 'Raipur', 'Ranchi', 'Nagpur',
          'Kimberley', 'Bloemfontein']

teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Punjab Kings', 'Rajasthan Royals', 'Mumbai Indians',
         'Delhi Capitals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad', 'Lucknow Super Giants',
         'Gujarat Titans']

img = get_img_as_base64("ipl.png")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None

    if request.method == 'POST':
        try:
            batting_team = request.form.get('batting_team')
            bowling_team = request.form.get('bowling_team')
            selected_city = request.form.get('city')
            target = int(request.form.get('target'))
            score = int(request.form.get('score'))
            overs = float(request.form.get('overs'))
            wickets = int(request.form.get('wickets'))

            runs_left = target - score
            balls_left = 120 - (overs * 6)
            wickets_remaining = 10 - wickets
            crr = score / overs
            rrr = runs_left / (balls_left / 6)

            input_data = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_remaining': [wickets_remaining],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            result = pipe.predict_proba(input_data)
            loss = result[0][0]
            win = result[0][1]
            result = {
                'batting_team': batting_team,
                'bowling_team': bowling_team,
                'win_percentage': round(win * 100),
                'loss_percentage': round(loss * 100)
            }
        except Exception as e:
            error = f"Some error occurred: {e}. Please check your inputs."

    return render_template('index.html', teams=teams, cities=cities, result=result, error=error, img=img)


if __name__ == '__main__':
    app.run(debug=False)


