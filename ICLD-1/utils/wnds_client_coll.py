from openai import OpenAI


def setOpenAi(keyid = 0):
        # set your openai key here.
        if keyid == 0:
            # base_url="https://api.siliconflow.cn"
            # api_key = "sk-jltwxwxwtyknufxsmprabvoqunyfcjyvwzwgzwxxaqubrchw"
            # base_url="https://openrouter.ai/api/v1"
            # api_key = "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"
            api_key="qwen",
            base_url="http://219.222.20.79:32520/"       #  qwen3-30b-a3b-instruct-2507-q5km
        client = OpenAI(base_url = base_url, api_key=api_key)
        return client


if __name__ == "__main__":

    client = setOpenAi()

    model = " "
    messages=[
        {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
        {"role": "user", "content": "Tell me what is AI?"}
    ]
    temperature = 0
    max_tokens = 500


    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    answer = response.choices[0].message.content

    print("=================================================")
    print("answer:", answer)
    print("=================================================")