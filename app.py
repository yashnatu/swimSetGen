import gradio as gr
from transformers import pipeline

# Load the model pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Generate a swim workout based on user input
def generate_swim_set(level, goal, strokes, duration):
    prompt = (
        f"Create a {duration}-minute swim workout for a {level} swimmer. "
        f"The focus should be on {goal}. Preferred strokes are: {strokes}. "
        "Include a detailed warm-up, a main set with rest intervals, and a cool-down. "
        "Format the workout clearly with bullet points or section headers."
    )
    result = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.8)[0]["generated_text"]
    return result

# Define Gradio interface
iface = gr.Interface(
    fn=generate_swim_set,
    inputs=[
        gr.Dropdown(["beginner", "intermediate", "advanced"], label="Swimmer Level"),
        gr.Textbox(lines=1, placeholder="e.g. endurance, speed, technique", label="Goal"),
        gr.Textbox(lines=1, placeholder="e.g. freestyle, IM, breaststroke", label="Preferred Strokes"),
        gr.Slider(20, 90, value=45, step=5, label="Workout Duration (minutes)")
    ],
    outputs=gr.Textbox(label="Generated Swim Workout"),
    title="🏊 Personalized Swim Set Generator",
    description="Generate a custom swim workout based on your experience, goals, and preferences using an AI model."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
