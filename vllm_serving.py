from vllm import LLM, SamplingParams
from vllm.entrypoints.openai import OpenAIServingEndpoint

# 파인튜닝된 모델 경로
model_path = "./finetuned_model"

# vLLM 모델 초기화
llm = LLM(model=model_path, trust_remote_code=True)

# 추론 파라미터 설정
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256
)

# 예시 프롬프트로 테스트
prompt = "Your test prompt here"
outputs = llm.generate([prompt], sampling_params)

# 결과 출력
for output in outputs:
    print(output.text)

# API 서버로 실행하기 (선택사항)
server = OpenAIServingEndpoint(model_path)
server.serve(host="0.0.0.0", port=8000)