from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and reference data
model = joblib.load("model.pkl")
X_columns = joblib.load("X_columns.pkl")
df_reference = pd.read_csv("visionspring_inventory_forecasting_data.csv")


def predict_frames_and_inventory_by_location(location_name, model, X_columns, df_reference):
    input_row = pd.DataFrame(columns=X_columns)
    input_row.loc[0] = 0

    input_row.at[0, 'Population'] = df_reference['Population'].mean()
    input_row.at[0, 'Camp_Duration_Days'] = df_reference['Camp_Duration_Days'].mean()
    input_row.at[0, 'Staff_Count'] = df_reference['Staff_Count'].mean()
    input_row.at[0, 'People_Screened'] = df_reference['People_Screened'].mean()
    input_row.at[0, 'Prescriptions_Given'] = df_reference['Prescriptions_Given'].mean()
    input_row.at[0, 'Frames_Stock_Start'] = df_reference['Frames_Stock_Start'].mean()
    input_row.at[0, 'Frames_Stock_End'] = df_reference['Frames_Stock_End'].mean()
    input_row.at[0, 'Month'] = 7
    input_row.at[0, 'Season_Monsoon'] = 1

    loc_col = f"Location_{location_name}"
    if loc_col in X_columns:
        input_row.at[0, loc_col] = 1
    else:
        raise ValueError(f"Location '{location_name}' not found.")

    ru_cols = [col for col in X_columns if col.startswith("Rural_Urban_")]
    if ru_cols:
        input_row.at[0, ru_cols[0]] = 1

    state_cols = [col for col in X_columns if col.startswith("State_")]
    if state_cols:
        input_row.at[0, state_cols[0]] = 1

    predicted_frames = int(round(model.predict(input_row)[0]))
    estimated_stock = int(round(input_row.at[0, 'Frames_Stock_Start']))

    return predicted_frames, estimated_stock


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        location = data.get("location")
        if not location:
            return jsonify({"error": "Missing location parameter"}), 400

        predicted_frames, estimated_stock = predict_frames_and_inventory_by_location(
            location_name=location,
            model=model,
            X_columns=X_columns,
            df_reference=df_reference
        )

        return jsonify({
            "location": location,
            "predicted_frames": predicted_frames,
            "estimated_stock": estimated_stock
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)