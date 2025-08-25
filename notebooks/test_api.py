from openai import OpenAI

client = OpenAI(
  api_key="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"
)
print("start")
response = client.responses.create(
  model="gpt-4o-mini",
  input="Tell me why you are so huai!",
  store=True,
)
print("end")
print(response.output_text);
