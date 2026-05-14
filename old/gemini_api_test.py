import google.generativeai as genai

# 配置模型
genai.configure(api_key="AIzaSyCzkslAayYsefjbNdvJwYWXQCX5J0PsLCY")

# 设置生成参数
generation_config = genai.types.GenerationConfig(
    candidate_count=1,           # 生成候选数量
    max_output_tokens=2048,      # 最大输出token数
    temperature=0.7,             # 创造性控制 (0-1)
    top_p=0.8,                  # 核采样参数
    top_k=40                    # Top-K采样参数
)

# 安全设置
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH", 
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# 初始化模型
model = genai.GenerativeModel(
    model_name='gemini-2.5-pro',
    generation_config=generation_config,
    safety_settings=safety_settings
)

# 生成内容
prompt = "请详细解释深度学习的工作原理"
response = model.generate_content(prompt)
print(response.text)

