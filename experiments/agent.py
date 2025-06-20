import uuid
from typing import Literal
from langgraph.prebuilt import create_react_agent
from langsmith import Client
from langchain_core.messages import ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict

client = Client()
# TODO: dataset name
dataset = "about-langgraph-3"

# TODO: initial instructions and a prompt name
initial_instructions = """Return whether or not this conversation is about langgraph"""

template = ChatPromptTemplate([
    {"role": "system", "content": "{{instructions}}"},
    {"role": "user", "content": "{{input.outputs.messages}}"},
])
prompt_handle = f"about-langgraph-react-{str(uuid.uuid4())[:8]}"
client.push_prompt(prompt_handle, object=template.partial(instructions=initial_instructions))
examples = {ex.id: ex for ex in client.list_examples(dataset_name=dataset)}


class Classification(TypedDict):
    about_langgraph: bool
    """Whether the conversation was about LangGraph."""

llm = init_chat_model("openai:gpt-4.1").with_structured_output(Classification)

def create_llm_judge(instructions: str):
    prompt = template.partial(instructions=instructions)
    return prompt | llm

def correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs.get('about_langgraph') == reference_outputs['about_langgraph']

def evaluate_prompt(new_prompt: str) -> str:
    """Evaluate a new prompt candidate against the dataset."""
    llm_judge = create_llm_judge(new_prompt)
    results = client.evaluate(
        llm_judge,
        data=dataset,
        evaluators=[correct],
    )
    df = results.to_pandas()
    
    return f"Experiment name: {results.experiment_name}\n\n{df.to_markdown()}"
    

def read_experiment_results(experiment_name: str, filter: Literal['correct', 'incorrect', 'all'] = 'all', limit: int = 5, offset: int = 5) -> list[dict]:
    """Read the experiment results."""
    if filter == 'correct':
        filter = "and(eq(feedback_key, 'correct'), eq(feedback_score, 1))"
    elif filter == 'incorrect':
        filter = "and(eq(feedback_key, 'correct'), eq(feedback_score, 0))"
    else:
        filter = None
    runs = list(client.list_runs(project_name=experiment_name, filter=filter))[offset: offset + limit]
    feedbacks = client.list_feedback(run_ids=[run.id for run in runs])
    feedbacks_by_run = {}
    for f in feedbacks:
        feedbacks_by_run.setdefault(f.run_id, []).append({f.key: f.score})
    return[
        {
            "inputs": ex[run.reference_example_id].inputs, 
            "outputs": run.outputs, 
            "expected_outputs": ex[run.reference_example_id].outputs,
            "feedback": feedbacks_by_run[run.id],
        } for run in runs
    ]

example_dicts = [ex.dict() for ex in examples.values()]

def read_dataset(limit: int = 5, offset: int = 0) -> list[dict]:
    """Get the examples in the dataset. Examples can be very long so don't look at all of them at once."""
    return example_dicts[offset:offset + limit]

def commit_prompt(new_prompt: str) -> str:
    """Commit the latest version of the prompt if it is an improvement over the current prompt."""
    return client.push_prompt(prompt_handle, template.partial(instructions=new_prompt))


def think(thought_process: str) -> None:
    """Use this to carefully think about how you want to update the prompt before running another evaluation."""
    return



def trim_messages(state):
    messages = []
    found_last = False
    for i, msg in enumerate(state['messages'][::-1]):
        if isinstance(msg, ToolMessage) and msg.name == 'read_experiment_results':
            if found_last:
                msg = msg.model_copy()
                # Drop the actual experiment results, just include the experiment name.
                msg.content = msg.content.split("\n")[0]
            else:
                found_last = True
            
        messages.append(msg)
    
    return {"llm_input_messages": list(reversed(messages))}

# add agent instructions / edit the first message.
agent = create_react_agent(
    "anthropic:claude-sonnet-4-20250514",
    tools=[
        evaluate_prompt,
        read_experiment_results,
        # read_dataset,
        commit_prompt,
        think,
    ],
    prompt="""You are the best prompt engineer in the world. \
    Your job is to optimize a prompt given a dataset. \
    You must evaluate your prompt updates against the dataset. \
    Every time you find a prompt that improves the performance on the dataset, \
    please commit that version of the prompt. Continue on until you think you cannot \
    improve the performance of the prompt anymore.""",
    pre_model_hook=trim_messages,
)


first_msg = f"""Optimize the following prompt:

```
{initial_instructions}
```

The evaluation dataset for this prompt has {len(examples)} examples.
"""

for chunk in agent.stream({"messages": [{"role": "user", "content": first_msg}]}):
    print(chunk)