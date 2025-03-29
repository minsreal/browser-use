import os

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent

load_dotenv()

api_key_deepseek = os.getenv('DEEPSEEK_API_KEY', '')
if not api_key_deepseek:
	raise ValueError('DEEPSEEK_API_KEY is not set')


async def run_agent(task: str, max_steps: int = 38):
	llm = ChatOpenAI(
		base_url='https://ark.cn-beijing.volces.com/api/v3',
		model='ep-20250206135139-nx2t5',
		api_key=SecretStr(api_key_deepseek),
	)
	agent = Agent(task=task, llm=llm, use_vision=False)
	result = await agent.run(max_steps=max_steps)
	return result


if __name__ == '__main__':
	task = '进入deepseek官网，查询v3模型的价格'
	result = asyncio.run(run_agent(task))
	print(result)