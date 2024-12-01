# Dependencies
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def getPrediction(filename):
    # Load the model
    model = load_model(r"C:\Users\Bhuvan M\OneDrive\Desktop\PROJECT\Resources\Model\final_model.hdf5")
    
    # Load and preprocess the image
    img_path = r'C:\Users\Bhuvan M\OneDrive\Desktop\PROJECT\static\\' + filename
    img = load_img(img_path, target_size=(180, 180))  # Resize image to the model's expected input size
    img = img_to_array(img)                          # Convert the image to a NumPy array
    img = img / 255.0                                # Normalize pixel values to the range [0, 1]
    img = np.expand_dims(img, axis=0)                # Add batch dimension (1, height, width, channels)
    
    # Predict the class and probabilities
    probabilities = model.predict(img)              # Returns probabilities for each class
    predicted_class = np.argmax(probabilities, axis=1)[0]  # Get the class index with the highest probability
    
    # Map class indices to labels
    if predicted_class == 1:
        answer = "Recycle"
        probability_result = probabilities[0][1]    # Probability of "Recycle"
    else:
        answer = "Organic"
        probability_result = probabilities[0][0]    # Probability of "Organic"

    # Prepare results
    answer = str(answer)
    probability_result = str(probability_result)
    values = [answer, probability_result, filename]
    
    return values[0], values[1], values[2]

