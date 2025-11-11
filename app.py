from flask import Flask, jsonify, render_template, request

import predictor

app = Flask(__name__)


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submissions and return a prediction."""
    data = request.get_json()

    home_team = data.get("homeTeam")
    away_team = data.get("awayTeam")

    if not home_team or not away_team:
        return jsonify({"error": "Both teams must be selected"}), 400

    home_win = predictor.predict_winner(home_team, away_team)

    return jsonify({"winner": home_team if home_win else away_team})


if __name__ == "__main__":
    app.run(debug=True)
