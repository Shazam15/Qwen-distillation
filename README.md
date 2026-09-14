# What is this for?

This repo was created with the purpose of distilling the Qwen 3.8 Flash model into Qwen 3, both 4b and 14b 

This is for the purpose of supercharging the smaller Qwen 3 to be used in the academic ai agent ATLAS.

The objective is to improve the answer quality of the locally served Qwen3 model used by ATLAS via response-based knowledge distillation, without introducing recurring API costs.

# What's distillation?

Distillation is basically taking a large LLM and using it to train a smaller one to act like it. Think of it them teacher and student. The student will always try to match what the teacher does. 

# Hardware and constraints

|Component|Spec|Role in the plan|
|---------|----|----------------|
|GPU|Tesla T4, 16GB VRAM (Turing, fp16 only)|QLora fine-tune of the student|
|CPU|Intel Xeon Gold 6258R|Runs the large teacher model for 
|RAM|~400GB|Holds the quantized teacher (Qwen3.8-Flash-Next~65-75GB in Q4) plus generation overhead|
|Serving layer|Ollama|Every LLM call site in ATLAS reads settings.LLM_MODEL as a plain string passed to ChatOllama. The distilled model is a deop-in tag swap, no code change needed|