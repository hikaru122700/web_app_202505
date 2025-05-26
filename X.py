import streamlit as st

st.title("Track Job Python Scripts")

post_content = st.text_input("Enter the content of your post:")
st.balloons()
if st.button("Submit"):
    if post_content:
        st.success("Post submitted successfully!")
        st.write("Your post content:")
        st.write(post_content)
    else:
        st.error("Please enter some content before submitting.")