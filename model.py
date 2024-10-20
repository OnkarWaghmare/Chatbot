from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def chat():
    chat_history_ids = None
    print("chat with the bot (type 'quit' to stop)!")
    while True:
        user_input = input(">> You: ")

        if user_input.lower() == "quit":
            break

        #encode user input and add end of string token
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # append the new user input to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

        #generate a response 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        #deocde the response 
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        print(f"<< Bot:{bot_response}")

#run the chat function
chat()


        