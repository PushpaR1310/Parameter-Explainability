from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Import SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Explainability features will be disabled.")

app = Flask(__name__, template_folder='templates')
app.config['ENV'] = 'production'
app.config['DEBUG'] = False

model = None
explainer = None

# Feature names in the order used by the model
FEATURE_NAMES = [
    'age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
    'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane'
]

# Human-readable feature names
FEATURE_DISPLAY_NAMES = {
    'age': 'Age',
    'bp': 'Blood Pressure',
    'al': 'Albumin Level',
    'su': 'Sugar Level',
    'rbc': 'Red Blood Cells',
    'pc': 'Pus Cell',
    'pcc': 'Pus Cell Clumps',
    'ba': 'Bacteria',
    'bgr': 'Blood Glucose Random',
    'bu': 'Blood Urea',
    'sc': 'Serum Creatinine',
    'pot': 'Potassium',
    'wc': 'White Blood Cell Count',
    'htn': 'Hypertension',
    'dm': 'Diabetes Mellitus',
    'cad': 'Coronary Artery Disease',
    'pe': 'Pedal Edema',
    'ane': 'Anemia'
}

def load_model() -> None:
    global model, explainer
    model = None
    explainer = None
    candidate_paths = ['kidney_new.pkl', 'kidney.pkl']
    last_error = None
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'rb') as file:
                loaded = pickle.load(file)
            if not hasattr(loaded, 'predict'):
                raise ValueError(f"Loaded object from {path} has no predict method")
            model = loaded
            print(f"✅ Model loaded successfully from {path}!")
            print(f"Model type: {type(model)}")
            print(f"Model has predict method: {hasattr(model, 'predict')}")
            
            # Initialize SHAP explainer if available
            if SHAP_AVAILABLE and model is not None:
                try:
                    # Use TreeExplainer for tree-based models (RandomForest)
                    explainer = shap.TreeExplainer(model)
                    print("✅ SHAP explainer initialized successfully!")
                except Exception as e:
                    print(f"⚠️ Could not initialize SHAP explainer: {e}")
                    explainer = None
            return
        except Exception as e:  # intentional: show precise import/ABI issues
            last_error = e
            print(f"❌ Error loading model from {path}: {e}")
    if model is None:
        print("❌ No usable model loaded.")
        if last_error is not None:
            print(f"Last load error: {last_error}")
        print("Tip: Train a new model (kidney_new.pkl) or fix dependency versions.")

load_model()

import numpy as _np
import sklearn as _sk
print("NumPy version:", _np.__version__)
print("scikit-learn version:", _sk.__version__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'model_loaded': model is not None,
        'model_type': str(type(model)) if model is not None else None,
        'numpy_version': _np.__version__,
        'sklearn_version': _sk.__version__,
    })

@app.route('/reload_model', methods=['POST'])
def reload_model():
    load_model()
    return jsonify({'success': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model is not available. Please restart the application.'
        }), 500
    
    try:
        
        data = request.get_json()
        
        
        all_features = [
            float(data['age']),
            float(data['bp']),
            float(data['sg']),
            int(data['al']),
            int(data['su']),
            float(data['bgr']),
            float(data['bu']),
            float(data['sc']),
            float(data['sod']),
            float(data['pot']),
            float(data['hemo']),
            int(data['pcv']),
            float(data['rc']),
            int(data['rbc']),
            int(data['pc']),
            int(data['pcc']),
            int(data['ba']),
            int(data['wc']),
            int(data['htn']),
            int(data['dm']),
            int(data['cad']),
            int(data['appet']),
            int(data['pe']),
            int(data['ane'])
        ]
        
        
        features = [
            all_features[0],  
            all_features[1],   
            all_features[3],   
            all_features[4],  
            all_features[13], 
            all_features[14],  
            all_features[15],  
            all_features[16], 
            all_features[5],  
            all_features[6],   
            all_features[7],   
            all_features[9],   
            all_features[17],  
            all_features[18],  
            all_features[19],  
            all_features[20],  
            all_features[22],  
            all_features[23]   
        ]
        
      
        features_array = np.array(features).reshape(1, -1)
        
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        result = "CHRONIC KIDNEY DISEASE (CKD)" if prediction == 1 else "NO CHRONIC KIDNEY DISEASE"
        confidence = max(probability) * 100
        
        # Calculate SHAP values for explainability
        feature_contributions = []
        if SHAP_AVAILABLE and explainer is not None:
            try:
                shap_values = explainer.shap_values(features_array)
                # For binary classification, shap_values is a list [values_for_class_0, values_for_class_1]
                # We want the values for the predicted class
                if isinstance(shap_values, list):
                    shap_vals = shap_values[prediction]  # Get values for predicted class
                else:
                    shap_vals = shap_values[0]
                
                # Create feature contributions list
                for i, feature_name in enumerate(FEATURE_NAMES):
                    contribution = float(shap_vals[0][i])
                    feature_contributions.append({
                        'feature': feature_name,
                        'display_name': FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
                        'contribution': round(contribution, 4),
                        'value': float(features[i])
                    })
                
                # Sort by absolute contribution (most influential first)
                feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            except Exception as e:
                print(f"⚠️ Error calculating SHAP values: {e}")
                feature_contributions = []
        else:
            # Fallback: use feature importance from the model if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, feature_name in enumerate(FEATURE_NAMES):
                    feature_contributions.append({
                        'feature': feature_name,
                        'display_name': FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
                        'contribution': round(float(importances[i]), 4),
                        'value': float(features[i])
                    })
                feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
       
        response = {
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'prediction': int(prediction),
            'recommendation': "Please consult a nephrologist immediately for proper diagnosis and treatment." if prediction == 1 else "Continue with regular health checkups and maintain a healthy lifestyle.",
            'feature_contributions': feature_contributions
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
