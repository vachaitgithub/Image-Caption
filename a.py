# import streamlit as st
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch

# # Set page configuration
# st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")

# # Load the Blip model and processor
# @st.cache_resource
# def load_model():
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     return processor, model

# processor, model = load_model()

# # Streamlit app layout
# st.title("🖼️ Image Caption Generator")
# st.markdown("""
# Welcome to the Image Caption Generator powered by the Blip model from Hugging Face! 
# Upload an image, and let our advanced AI generate a descriptive caption for you. 
# Perfect for content creators, marketers, and enhancing accessibility! 🚀
# """)

# # Image upload section
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Generate caption
#     with st.spinner("Generating caption..."):
#         # Process the image
#         inputs = processor(images=image, return_tensors="pt")
        
#         # Generate caption
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_length=50, num_beams=5)
#         caption = processor.decode(outputs[0], skip_special_tokens=True)
        
#         # Display the caption
#         st.success("**Generated Caption:**")
#         st.write(caption)
# else:
#     st.info("Please upload an image to generate a caption.")

# # Footer
# st.markdown("""
# ---
# **Powered by**: Hugging Face Blip Model | Built with Streamlit  
# *Enhance your visual content with AI-driven captions!* ✨
# """)









import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import urllib.parse

# Set page configuration
st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")

# Load the Blip model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Streamlit app layout
st.title("🖼️ Image Caption Generator")
st.markdown("""
Welcome to the Image Caption Generator powered by the Blip model from Hugging Face! 
Upload an image, and let our advanced AI generate a descriptive caption for you. 
Perfect for content creators, marketers, and enhancing accessibility! 🚀
""")

# Image upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    with st.spinner("Generating caption..."):
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate caption
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_beams=5)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Create a Google search URL for the caption
        search_query = urllib.parse.quote(caption)
        search_url = f"https://www.google.com/search?q={search_query}"
        
        # Display the clickable caption
        st.success("**Generated Caption:**")
        st.markdown(f"[{caption}]({search_url})")
else:
    st.info("Please upload an image to generate a caption.")

# Footer
st.markdown("""
---
**Powered by**: Hugging Face Blip Model | Built with Streamlit  
*Enhance your visual content with AI-driven captions!* ✨
""")