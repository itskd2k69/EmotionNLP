import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Manual mapping of label indices to emotion names
label_map = {
        0: ('joy', '😄'),
    1: ('sadness', '😢'),
    2: ('anger', '😠'),
    3: ('love', '❤️'),
    4: ('fear', '😨'),
    5: ('surprise', '😲')
}

# UI
st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("💬 Emotion Prediction from Text")
st.markdown("Enter a sentence and find out the **emotion** behind it.")

user_input = st.text_area("📝 Enter your sentence:", height=150)

if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            X = vectorizer.transform([user_input])
            pred_index = model.predict(X)[0]  # numeric label
            emotion = label_map.get(pred_index, "Unknown")  # convert number to text label

            st.success(f"🎯 **Predicted Emotion:** `{emotion.upper()}`")
        except Exception as e:
            st.error(f"❌ Error during prediction:\n\n{e}")
