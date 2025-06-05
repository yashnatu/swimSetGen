from transformers import pipeline

# Load a text-generation pipeline with a lightweight model
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Define your prompt
prompt = (
    "Create a 45-minute swim workout for an intermediate swimmer. "
    "They want to focus on endurance and prefer freestyle and IM. "
    "Include a detailed warm-up, a challenging main set with rest intervals, "
    "and a relaxing cool-down. Format the workout clearly."
)

# Generate the response
response = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)

# Print the generated swim workout
print("🏊 Generated Swim Set:\n")
print(response[0]['generated_text'])
