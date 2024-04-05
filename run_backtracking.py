"""Script to run end-to-end evaluation on the benchmark"""
import argparse
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
import copy

import openai

from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_image_tag_action,
)

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    generate_from_4v_chat_completion,
    lm_config,
)

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

class Summarizer():
    def __init__(self, URL: str, accessibility_tree: str, instruction_path: str, model: str = "gpt-3.5-turbo"):
        self.URL = URL
        self.accessibility_tree = accessibility_tree
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.model = model
    
    def construct_prompt(self):
        URL = self.URL
        accessibility_tree = self.accessibility_tree
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]

        current = template.format(
            URL=URL,
            accessibility_tree=accessibility_tree,
        )

        message = [{"role": "system", "content": intro}]
        for (x, y) in examples:
            message.append(
                {
                    "role": "system",
                    "name": "example_user",
                    "content": x,
                }
            )
            message.append(
                {
                    "role": "system",
                    "name": "example_assistant",
                    "content": y,
                }
            )
        message.append({"role": "user", "content": current})
        return message
    
    def call_llm(self, prompt):
        response = generate_from_openai_chat_completion(
                messages=prompt,
                model=self.model,
                stop_token=None,
            )

    def summarize(self) -> str:
        prompt = self.construct_prompt()
        response = self.call_llm(prompt)
        return response

class Backtracker():
    def __init__(self, env: ScriptBrowserEnv, instruction_path: str, objective: str, model: str = "gpt-4-turbo-preview"):
        self.env = env
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.objective = objective
        self.model = model

        # form a list of (accessibility_tree, URL) pairs
        # trajectory here is a list of interleaved observation and action, ended with a stop action
        # self.state_url_pairs = [(state_info["observation"]["text"], state_info["info"]["page"].url) for state_info in trajectory[::2]]

    # def summarize_state(URL: str, accessibility_tree: str):
    #     # summarize the current state
    #     summarizer = Summarizer(URL, accessibility_tree, "agents/prompts/state_action_agent.json")
    #     summary = summarizer.summarize()
    #     return summary
    
    def construct_history(self, trajectory: Trajectory, metadata: dict[str, Any]):
        # history should be a list with [summarized_state, action_str] lists
        history = []
        state_summary_dict = metadata["state_summary_dict"]
        action_history = metadata["action_history"]
        for i in range(len(trajectory)):
            if i % 2 == 0: # observation
                state = trajectory[i]["observation"]["text"]
                summary = state_summary_dict.get(state, "Summary of the page not available")
                history.append([summary])
            else: # action
                # get action_str from action_history
                action_str = action_history[(i-1)//2]
                history[-1].append(action_str)
        return history

    def construct_prompt(self, history: list[list[str]]) -> str:
        accessibility_tree = self.accessibility_tree
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        objective = self.objective

        """
        construct the history string like: STATE 0: [state_summary], ACTION 0: [action_str]...
        """
        history_str = ""
        for i, his in enumerate(history):
            state_summary = his[0]
            action_str = his[1]
            history_str += f"STATE {i}: {state_summary}, ACTION {i}: {action_str}\n"

        current = template.format(
            history = history_str,
            objective = objective,
        )

        message = [{"role": "system", "content": intro}]
        for (x, y) in examples:
            message.append(
                {
                    "role": "system",
                    "name": "example_user",
                    "content": x,
                }
            )
            message.append(
                {
                    "role": "system",
                    "name": "example_assistant",
                    "content": y,
                }
            )
        message.append({"role": "user", "content": current})
        return message

    def call_llm(self, prompt: str):
        response = generate_from_openai_chat_completion(
                messages=prompt,
                model=self.model,
                stop_token=None,
            )
        return response

    def parse_response(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f'Parsing error for backtracking response: {response}'
            )
        
    def act(self, trajectory: Trajectory, metadata: dict[str, Any]):
        # c here means constant which will consider the "return" action taken by the backtracker
        c_trajectory = copy.deepcopy(trajectory)
        c_metadata = copy.deepcopy(metadata)

        history = self.construct_history(trajectory=trajectory, metadata=metadata)
        prompt = self.construct_prompt(history)
        response = self.call_llm(prompt)
        parsed_res = self.parse_response(response)

        # ground the action to the env
        # one example of the parsed_res is "return [1]" which means return to the 1st state (starting from 0)

        state_num = int(parsed_res.split()[1])
        # find the state_info from the trajectory
        state_info = trajectory[state_num*2]
        
        # build an action of go_to [URL] to return to the state
        URL = state_info["info"]["page"].url
        action = create_id_based_action(f"goto [{URL}]")
        obs, _, terminated, _, info = self.env.step(action)

        # update the constant trajectory and metadata by appending the return action and obs to the trajectory
        c_trajectory.append(action)
        c_trajectory.append({"observation": obs, "info": info})
        c_metadata["action_history"].append(f"return to state {state_num}")

        # update the trajectory by removing the states and actions after the state_num-th state
        new_trajectory = trajectory[:state_num*2+1]

        # update the action history in metadata by removing the actions after the state_num-th action_str
        metadata["action_history"] = metadata["action_history"][:state_num]
        new_metadata = metadata

        return (obs, new_trajectory, new_metadata, c_trajectory, c_metadata)

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    # here, for set of mark prompting, we need to use seperate grounding method, so here we distinguish it from the other action sets
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--selected_files",
        default=[],
        nargs="+",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--record_dir", default="records", type=str)

    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args

# helper function to save np array to image
def save_np_to_image(np_array: np.ndarray, path: str)->None:
    from PIL import Image
    image = Image.fromarray(np_array)
    # Extract directory from the provided path
    dir_path = os.path.dirname(path)
    # Check whether directory exists, if not, create it
    if not os.path.exists(dir_path) and dir_path != '':
        os.makedirs(dir_path)
    image.save(path)
    return image

def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def test(
    args: argparse.Namespace,
    agent: Agent | PromptAgent | TeacherForcingAgent,
    config_file_list: list[str],
) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
    )

    for config_file in config_file_list:
        # parse the id from config_file: config_files/{id}.json
        id = os.path.basename(config_file).split(".")[0]
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # get intent
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            "python",
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            c_trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            step_count = 0
            # output STOP[N/A] times
            stop_na_count = 0

            # store the image locally and load it later into the prompt of gpt-4v
            try:
                image_path = f"{args.result_dir}/images/{id}/step_{step_count}.png"
                image = save_np_to_image(obs["image"], image_path)
                # change obs["image"] to image_path
                obs["image_path"] = image_path
            except Exception as e:
                print(e)

            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)
            c_trajectory.append(state_info)

            state = state_info["observation"]["text"]
            summary = Summarizer(state_info["info"]["page"].url, state, "agents/prompts/summarizer.json").summarize()
            print("Initial state summary: ", summary)

            # eliminated_s_a_dict is used to store the eliminated state-action pairs, where the key is the state (DOM here) and the value is the list of actions that have been taken before but failed
            # state_summary_dict is used to store the summarized state, where the key is the state (DOM here) and the value is the summarized state
            meta_data = {"action_history": ["None"], "eliminated_s_a_dict": {}, "state_summary_dict": {state: summary}}
            c_meta_data = {"action_history": ["None"], "eliminated_s_a_dict": {}, "state_summary_dict": {state: summary}}

            while True:
                step_count += 1
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        action = agent.next_action(
                            trajectory, intent, meta_data=meta_data, info=info
                        )
                    except ValueError as e:
                        # get the error message
                        action = create_stop_action(f"ERROR: {str(e)}")
                    
                print(action)

                '''
                example of action:
                {'action_type': , 'coords': array([0., 0.], dtype=float32), 'element_role': 31, 'element_name': '', 'text': [], 'page_number': 0, 'url': '', 'nth': 0, 'pw_code': '', 'element_id': '19', 'key_comb': '', 'direction': '', 'answer': '', 'raw_prediction': 'To find the top search terms in the store, I would typically look for a section in the dashboard that provides insights or analytics on customer behavior, which could include search term data. However, the screenshot provided does not show a direct link or section for search terms analytics on the current view of the dashboard.\n\nAdvanced reporting, which could potentially contain search term analytics, can sometimes be accessed through a link like the one indicated by ID [19]. Since there are no other obvious options, I will proceed by clicking on the "Go to Advanced Reporting" link to look for the top search terms.\n\nIn summary, the next action I will perform is ```click [19]```.', 'bounding_boxes': [{'left': 992.578125, 'top': 320.5625, 'right': 1203.1875, 'bottom': 342.5625, 'width': 210.609375, 'height': 22}]}
                '''

                trajectory.append(action)
                c_trajectory.append(action)
                
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)
                # update the eliminated_s_a_dict
                state = state_info["observation"]["text"]
                if state in meta_data["eliminated_s_a_dict"]:
                    meta_data["eliminated_s_a_dict"][state].append(action_str)

                # if the action is stop, check whether it is an N/A
                answer = action.get("answer", "")
                if action["action_type"] == ActionTypes.STOP:
                    if answer == "N/A":
                        # call another module to deal with the N/A case
                        obs, trajectory, meta_data, c_trajectory, c_meta_data = Backtracker(trajectory, env, meta_data).act(trajectory=trajectory, metadata=meta_data)

                        # render this special part
                        render_helper.render(
                            action, state_info, c_meta_data, args.render_screenshot
                        )

                        # increment the stop_na_count
                        stop_na_count += 1
                        # check whether the stop_na_count exceeds the threshold, if so, break the loop and use N/A as the answer
                        if stop_na_count > 3:
                            break
                    else:
                        break

                # the coord is a numpy array, which is not json serializable so we need to convert it to list
                for key, value in action.items():
                    if isinstance(value, np.ndarray):
                        action[key] = value.tolist()

                try:
                    print(f"Step {step_count} action: {action}")
                    obs, _, terminated, _, info = env.step(action)
                    image_path = f"{args.result_dir}/images/{id}/step_{step_count}.png"
                    image = save_np_to_image(obs["image"], image_path)
                    obs["image_path"] = image_path
                except Exception as e:
                    terminated = False
                    print(e)
                
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)
                c_trajectory.append(state_info)

                state = state_info["observation"]["text"]
                if state not in meta_data["state_summary_dict"]:
                    summary = Summarizer(state_info["info"]["page"].url, state, "agents/prompts/summarizer.json").summarize()
                    meta_data["state_summary_dict"][state] = summary                

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    c_trajectory.append(create_stop_action(""))
                    break

            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

        except openai.error.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        render_helper.close()

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    args = config()
    args.sleep_after_execution = 2.0
    prepare(args)

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx

    # add selected files flag
    if args.selected_files != []:
        str_file_idxs = args.selected_files
        file_idxs = [int(idx) for idx in str_file_idxs]
        print(f"Selected {len(file_idxs)} tasks: {file_idxs}")
    else:
        file_idxs = range(st_idx, ed_idx)
    
    for i in file_idxs:
        test_file_list.append(f"config_files/{i}.json")

    if "debug" not in args.result_dir:
        test_file_list = get_unfinished(test_file_list, args.result_dir)

    if len(test_file_list) == 0:
        logger.info("No task left to run")
    else:
        print(f"Total {len(test_file_list)} tasks left")
        args.render = False
        args.render_screenshot = True
        args.save_trace_enabled = True

        args.current_viewport_only = True
        dump_config(args)

        agent = construct_agent(args)
        test(args, agent, test_file_list)
