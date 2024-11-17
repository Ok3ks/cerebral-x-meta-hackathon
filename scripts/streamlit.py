import streamlit as st
import speech_recognition as sr

from agent_workflow import MyWorkflow

def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Recording...")
        audio = recognizer.listen(source)
        st.success("Recording complete.")
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

async def main():
    st.title("Text and Voice Input Application")

    # Text Input
    st.header("Text Input")
    text_input = st.text_area("Enter your text here:")

    # Display text input
    if text_input:
        w = MyWorkflow(timeout=10, verbose=False)
        result = await w.run(query=text_input)
        # st.write("Response")
        print(result)

    # Voice Input
    st.header("Voice Input")
    if st.button("Record Voice"):
        voice_input = transcribe_audio()
        st.write("Transcribed Voice Input:")
        st.write(voice_input)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())