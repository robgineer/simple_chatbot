import torch
import streamlit as st
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM

# set the title
st.title("🤖 Gemma3 Chatbot")
st.set_page_config(page_title="Gemma3 Chatbot")

# ------------------------------
# Load model
# ------------------------------
if "model" not in st.session_state:
    model_id = "google/gemma-3-1b-it"
    # use lower precision (this enables running the model on a smaller GPU)
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    # define model for streamlit
    st.session_state.model = Gemma3ForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
    ).eval()
    # define tokenizer for streamlit
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_id)

# ------------------------------
# Chat history
# ------------------------------
# set up chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# display chat history (user input and system response)
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# ------------------------------
# User input and inference
# ------------------------------
if prompt := st.chat_input("Ask me something"):
    # user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # tokenize using chat template
    input_ids = st.session_state.tokenizer.apply_chat_template(
        st.session_state.messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    # move to device
    input_ids = input_ids.to(st.session_state.model.device)

    # run inference
    with torch.inference_mode():
        outputs = st.session_state.model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
        )

    # decode response
    response = st.session_state.tokenizer.decode(
        outputs[0][input_ids.shape[1] :],
        skip_special_tokens=True,
    )

    # save and display response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
