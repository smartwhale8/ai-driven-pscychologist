import time
from openai import OpenAI

from google.colab import userdata
open_ai_key = userdata.get('OPENAI_API_KEY')

client = OpenAI(
    api_key=open_ai_key
)

psychologistAssistant = client.beta.assistants.create(
    name="Psychologist",
    instructions="If you are a licensed psychologist, please provide this patient with a helpful response to their concern..",
    model="gpt-3.5-turbo",
)

expertPsychologistEvalAssistant = client.beta.assistants.create(
    name="ExpertPsychologistEvaluator",
    instructions="You are an expert psychologist with extensive knowledge and experience in mental health. Your role is to evaluate the responses provided by a psychologist to a patient seeking help for mental health issues. When a psychologist's response is provided to you, you should critically analyze it and provide constructive feedback. Consider the following aspects: - Accuracy and correctness of the information provided, - Empathy and sensitivity towards the patient's concerns, - Clarity and understandability of the language used, Appropriateness and relevance of the advice or suggestions given, Any potential gaps or missing information that should be addressed. Your feedback should be professional, objective, and aimed at helping the psychologist improve their responses. Provide specific suggestions or recommendations for improvement, but avoid being overly critical or dismissive. Remember, your goal is to ensure that the patient receives high-quality, effective, and compassionate support from the psychologist.",
    model="gpt-3.5-turbo",
)

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="""I'm feeling really overwhelmed with work and family responsibilities. I don't know how to manage everything.""",
)

def runAssistant(assistant_id,thread_id,user_instructions):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=user_instructions,
    )

    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        if run.status == "completed":
            print("This run has completed!")

            break
        else:
            print("in progress...")
            time.sleep(5)

# Run the Writer Assistant to create a first draft
runAssistant(psychologistAssistant.id,thread.id,"Respond to the patient's concerns")

# Run the Critic Assistant to provide feedback
runAssistant(expertPsychologistEvalAssistant.id,thread.id,"""You are an expert psychologist with extensive knowledge and experience in mental health. Your role is to evaluate the responses provided by a psychologist to a patient seeking help for mental health issues. When a psychologist's response is provided to you, you should critically analyze it and provide constructive feedback. Consider the following aspects: - Accuracy and correctness of the information provided, - Empathy and sensitivity towards the patient's concerns, - Clarity and understandability of the language used, Appropriateness and relevance of the advice or suggestions given, Any potential gaps or missing information that should be addressed. Your feedback should be professional, objective, and aimed at helping the psychologist improve their responses. Provide specific suggestions or recommendations for improvement, but avoid being overly critical or dismissive. Remember, your goal is to ensure that the patient receives high-quality, effective, and compassionate support from the psychologist. Provide constructive feedback to what
the psychologistAgent assistant has mentioned""")

# Have the Writer Assistant rewrite the first chapter based on the feedback from the Critic
runAssistant(psychologistAssistant.id,thread.id,"""Using the feedback from the Expert Psychologist Evaluation Assistant provide a new resposne to the patient""")

messages = client.beta.threads.messages.list(
  thread_id=thread.id
)

for thread_message in messages.data:
    # Iterate over the 'content' attribute of the ThreadMessage, which is a list
    for content_item in thread_message.content:
        # Assuming content_item is a MessageContentText object with a 'text' attribute
        # and that 'text' has a 'value' attribute, print it
        print(content_item.text.value)