import google.generativeai as genai
import inspect

print("Gemini SDK version:", genai.__version__)
print("Gemini SDK file path:", inspect.getfile(genai))
