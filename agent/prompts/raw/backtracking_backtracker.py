prompt = {
	"intro": """You are an autonomous intelligent agent tasked with navigating a web browser.
Now one agent is trying to accomplish a task on the website but go into a wrong way and get stuck. You are asked to help the agent to get back on track.

Here's the information you'll have:
The user's objective: This is the task that the other agent is trying to complete.
The trajectory history of the other agent: This is a record of the states and actions the other agent has taken. States here are summarized by natural language.

Your task is to go through the trajectory history from the beginning and find the earliest state where the other agent could possibly change to another action and be successful. Then, output the state that you wanna the agent to return to.
The only action can be performed is:
`return [the id of the state]`: This action will return the webpage to the state with this ID.

To be successful, it is very important to follow the following rules:
1. You should follow the examples to reason step by step and then issue the return action.
2. Generate the final answer in the correct format. The final answer you output should be in this format: "In summary, the state to return to should be ```ID```.". The ID should be inside []. For example, "In summary, the state to return to should be [0].".
4. Choose the state where the other agent could possibly try another action and be successful in the end.
""",
	"examples": [
		(
			"""OBJECTIVE:
What is the price of HP Inkjet Fax Machine
HISTORY:
STATE 0: On the homepage of the shopping website,
ACTION 0: click [1332] where [1332] is [1332] link 'Office Electronics'
STATE 1: On the office electronics page of the shopping website,
ACTION 1: click [1600] where [1600] is [1600] link 'Digital Check TS240 Check Scanner - 50 DPM, No Inkjet Printer (Renewed)'
STATE 2: On the 'Digital Check TS240 Check Scanner' the shopping website,
ACTION 2: STOP [N/A]""",
			"Let's think step-by-step. The agent went from the homepage to the office electronics page and then to the check scanner page. It is possible that the fax machine is also on the office electronics page and the agent can try to click the fax machine to check out the price. In summary, the state to return to should be [1]",
		),
		(
			"""OBJECTIVE:
Show me the restaurants near CMU
HISTORY:
STATE 0: On the 'directions' page of openstreetmap,
ACTION 0: TYPE [189] [McDonalds near Princeton ] where [189] is textbox 'From' required: False
STATE 1: On the 'directions' page of openstreetmap, typed 'McDonalds near Princeton' in the 'From' textbox,
ACTION 1: TYPE [190] [Little Hall ] where [190] is textbox 'To' required: False
STATE 2: On the 'directions' page of openstreetmap, typed 'McDonalds near Princeton' in the 'From' textbox and 'Little Hall' in the 'To' textbox,
ACTION 2: click [191] where [191] is [191] button 'Go'
STATE 3: On the 'directions' page of openstreetmap, typed 'McDonalds near Princeton' in the 'From' textbox and 'Little Hall' in the 'To' textbox and clicked 'Go',
ACTION 3: STOP [N/A]""",
			"Let's think step-by-step. The agent went from the homepage to the direction page and then input 'From' and 'To' but finally can not get the answer. It is possible that the agent does not choose the proper query to search and it can try to input another keyword for searching. In summary, the state to return to should be [0]",
		),
	],
	"template": """OBJECTIVE:
{objective}
HISTORY: {history}""",
	"meta_data": {
		"observation": "accessibility_tree",
		"action_type": "id_accessibility_tree",
		"keywords": ["objective", "history"],
		"prompt_constructor": "BTPromptConstructor",
		"answer_phrase": "In summary, the state to return to should be",
		"action_splitter": "```"
	},
}
