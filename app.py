import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from openai import OpenAI

# 📌 Configuration of AI/ML API (⚠️ Replace with your API Key)
API_KEY = "7035f4f20430487a86ebe6445b694f20"
BASE_URL = "https://api.aimlapi.com/v1"

# 📌 Initialize OpenAI client with AI/ML API
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

# 📌 Load the pre-trained Moroccan food classification model
MODEL_PATH = "best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# 📌 List of Moroccan dishes (same as the dataset labels)
CLASS_NAMES = ['Sellou', 'Harcha', 'Sfenj', 'Batbout', 'Bastila', 'Loubia', 'Karan', 'Seffa', 'Tagine', 
               'Mechoui', 'Bessara', 'Khringo', 'Kaab el ghazal', 'Baghrir', 'Harira', 'Couscous', 'Fekkas', 
               'Matbucha', 'Msemen', 'Shakchouka', 'Rfissa', 'Maakouda', 'Briouat', 'Chebakia', 'Tanjia']

# 📌 Function to classify a Moroccan dish from an image
def predict_dish(image):
    """
    Processes the image and uses the trained model to predict the Moroccan dish.
    
    Parameters:
        image (PIL Image): The input image of the dish.
    
    Returns:
        predicted_dish (str): The name of the predicted Moroccan dish.
        confidence (float): The model's confidence in the prediction (0-100%).
    """
    img = image.resize((224, 224))  # Resize image to match the model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)  # Get predictions
    predicted_index = np.argmax(predictions[0])  # Get class index
    confidence = predictions[0][predicted_index] * 100  # Convert confidence score
    predicted_dish = CLASS_NAMES[predicted_index]  # Get class name

    return predicted_dish, confidence

# 📌 Function to generate a cultural description using AI/ML API
def generate_cultural_description(predicted_dish):
    """
    Calls AI/ML API to generate a short and rich cultural description of the Moroccan dish.
    
    Parameters:
        predicted_dish (str): The name of the Moroccan dish to describe.
    
    Returns:
        description (str): AI-generated cultural insight about the dish.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Moroccan cuisine and culture. Provide short, rich descriptions of Moroccan dishes."
                },
                {
                    "role": "user",
                    "content": f"Describe the Moroccan dish '{predicted_dish}'. Mention its cultural significance, main ingredients, and traditional preparation in less than 256 characters."
                },
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error while generating description: {e}"

# 📌 Moroccan Travel Journal (Session State)
if "travel_journal" not in st.session_state:
    st.session_state.travel_journal = []

def add_to_travel_journal(dish, note):
    """
    Adds a new entry to the user's Moroccan Travel Journal.
    
    Parameters:
        dish (str): The Moroccan dish that was experienced.
        note (str): User's personal note or experience.
    """
    st.session_state.travel_journal.append({"dish": dish, "note": note})

# 📌 Streamlit Web App Configuration
st.set_page_config(page_title="Moroccan Culinary Journey", layout="wide")

st.title("🇲🇦 Moroccan Culinary Journey 🍽️")
st.write("Discover the rich flavors and cultural heritage of Moroccan cuisine!")

# 📸 Image Upload or Capture Section
uploaded_file = st.file_uploader("Upload an image of a Moroccan dish", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Or take a photo")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif camera_image is not None:
    image = Image.open(camera_image)

if image is not None:
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # 🔍 Predict the Moroccan dish
    predicted_dish, confidence = predict_dish(image)
    st.subheader(f"🍛 Predicted Dish: {predicted_dish} ({confidence:.2f}% confidence)")
    
    # 📜 Generate Cultural Description using AI
    description = generate_cultural_description(predicted_dish)
    st.write(f"📜 **Cultural Insight:** {description}")

    # 📖 Add to Moroccan Travel Journal
    st.subheader("📖 Moroccan Travel Journal")
    user_note = st.text_area(f"Write your experience about {predicted_dish}:")

    if st.button("Save to My Travel Journal"):
        add_to_travel_journal(predicted_dish, user_note)
        st.success(f"{predicted_dish} added to your travel journal!")

    # 📝 Display Travel Journal Entries
    if st.session_state.travel_journal:
        st.subheader("📔 Your Moroccan Culinary Discoveries:")
        for entry in st.session_state.travel_journal:
            st.write(f"**{entry['dish']}** - {entry['note']}")

    # 🔗 Share Button (Simulation)
    share_button = st.button("📤 Share on Social Media")
    if share_button:
        st.success("🔗 Your experience has been prepared for sharing!")

# 🎨 Custom CSS Styling
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #f39c12;
            color: white;
            font-size: 16px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)
