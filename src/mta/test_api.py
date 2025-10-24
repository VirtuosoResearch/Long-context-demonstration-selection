from openai import OpenAI
client = OpenAI(api_key="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")

models = client.models.list()
for m in models.data:
    print(m.id)