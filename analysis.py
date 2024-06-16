import os
import re
import json

def analyze_errors():
    # analyze the parsing errors

    # read output files from PATH/render_{id}.html
    PATH = "/home/ytliu/github/webarena_update/outputs/gpt-4o-dom/result"

    # read the output files
    outputs = []
    for file in os.listdir(PATH):
        if file.endswith(".html"):
            with open(f"{PATH}/{file}", "r") as f:
                id = file.split("_")[1].split(".")[0]
                outputs.append((id, f.read()))

    parsing_error_ids = []
    repetitive_action_ids = []
    for (id, content) in outputs:
        # check parsing error
        if "Failed to parse actions" in content:
            parsing_error_ids.append(int(id))
        
        # check repetitive actions
        if "Same action" in content:
            repetitive_action_ids.append(int(id))

    # sort the ids by increasing order
    parsing_error_ids = sorted(parsing_error_ids)
    repetitive_action_ids = sorted(repetitive_action_ids)

    # convert ints back to strings
    parsing_error_ids = list(map(str, parsing_error_ids))
    repetitive_action_ids = list(map(str, repetitive_action_ids))

    num_parsing_errors = len(parsing_error_ids)
    num_repetitive_actions = len(repetitive_action_ids)

    parsing_error_ids_str = " ".join(parsing_error_ids)
    repetitive_action_ids_str = " ".join(repetitive_action_ids)

    print(f"Number of parsing errors: {num_parsing_errors}")
    print(f"Parsing error ids: {parsing_error_ids_str}")
    print(f"Number of repetitive actions: {num_repetitive_actions}")
    print(f"Repetitive action ids: {repetitive_action_ids_str}")

def get_pass_ids():
    PATH = "/home/ytliu/github/webarena_update/log_files/log_20240607091908_5880.log"

    # read the log file into a single string
    with open(PATH, "r") as f:
        log_data = f.read()

    # Regular expression pattern to find all PASS results
    pattern = re.compile(r'\[Result\] \(PASS\) .*/(\d+)\.json')

    # Find all matches
    pass_ids = pattern.findall(log_data)
    print(pass_ids)
    print(len(pass_ids))

def filter_tasks():
    path = "/home/ytliu/github/webarena_update/config_files"
    
    gitlab_ids = []
    # read the json files under the path one by one
    for file in os.listdir(path):
        if file.endswith(".json") and "test" not in file:
            # load the file
            with open(f"{path}/{file}", "r") as f:
                data = json.load(f)
                id = data["task_id"]
                sites = data["sites"]
                if "gitlab" in sites:
                    gitlab_ids.append(id)
    
    print(gitlab_ids)

def filter_results():
    gitlab_ids = [666, 749, 398, 568, 523, 447, 476, 791, 559, 340, 414, 570, 103, 751, 789, 45, 179, 132, 662, 527, 443, 317, 483, 421, 136, 755, 533, 578, 591, 205, 350, 296, 171, 687, 560, 390, 799, 802, 169, 180, 303, 175, 785, 418, 537, 806, 555, 307, 745, 564, 683, 394, 579, 389, 590, 170, 297, 391, 561, 686, 168, 803, 181, 349, 784, 293, 174, 536, 452, 419, 594, 554, 807, 306, 744, 395, 682, 565, 748, 667, 522, 446, 569, 558, 477, 341, 102, 415, 750, 312, 133, 178, 788, 44, 736, 663, 526, 442, 482, 316, 420, 106, 411, 754, 592, 206, 172, 295, 318, 563, 684, 393, 742, 801, 552, 176, 786, 357, 534, 450, 304, 658, 556, 805, 479, 448, 567, 397, 746, 669, 444, 665, 339, 343, 308, 475, 809, 752, 417, 46, 811, 310, 484, 688, 524, 661, 259, 422, 135, 314, 480, 756, 413, 577, 104, 445, 664, 309, 342, 808, 753, 416, 810, 485, 311, 525, 441, 156, 258, 660, 134, 481, 315, 105, 412, 576, 207, 593, 783, 294, 173, 392, 685, 562, 743, 182, 553, 800, 787, 177, 670, 535, 451, 659, 305, 804, 478, 557, 396, 566, 681, 449, 668, 747]
    
    success_case_ids_1 = ['8', '24', '36', '38', '39', '44', '47', '48', '70', '72', '73', '84', '88', '89', '90', '91', '92', '93', '94', '95', '96', '103', '115', '137', '139', '150', '151', '152', '155', '157', '162', '164', '168', '183', '187', '188', '189', '190', '192', '199', '208', '209', '231', '232', '233', '247', '248', '252', '253', '257', '258', '259', '260', '274', '275', '276', '278', '311', '320', '326', '357', '374', '389', '390', '394', '395', '396', '397', '410', '412', '419', '422', '426', '432', '435', '446', '453', '465', '466', '467', '469', '472', '478', '481', '482', '483', '484', '485', '512', '514', '515', '516', '517', '518', '576', '577', '579', '581', '594', '650', '662', '677', '678', '723', '739', '740', '742', '745', '757', '758', '772', '774', '775', '784', '785', '790', '795', '797', '799', '802', '803']

    # success case ids
    success_case_ids_2 = ['0', '8', '14', '22', '24', '27', '36', '38', '39', '44', '47', '48', '70', '71', '72', '73', '84', '88', '89', '90', '91', '92', '93', '94', '95', '96', '115', '126', '128', '132', '151', '152', '155', '156', '157', '164', '166', '168', '187', '188', '189', '190', '192', '199', '200', '205', '206', '209', '211', '230', '231', '232', '233', '237', '247', '248', '252', '253', '258', '259', '260', '274', '275', '276', '278', '303', '305', '306', '310', '311', '313', '326', '341', '348', '357', '358', '359', '362', '368', '374', '389', '390', '394', '395', '396', '397', '410', '412', '421', '426', '431', '432', '433', '434', '435', '436', '446', '447', '457', '465', '467', '468', '469', '472', '475', '476', '478', '481', '482', '483', '484', '485', '491', '511', '513', '515', '516', '517', '568', '576', '577', '579', '582', '588', '598', '602', '650', '662', '665', '670', '680', '690', '704', '710', '712', '713', '723', '737', '739', '740', '741', '742', '745', '757', '758', '772', '783', '784', '787', '799', '802', '803']

    # convert the success case ids to integers
    success_case_ids_1 = list(map(int, success_case_ids_1))

    # get the success_case_ids that are not in gitlab_ids
    missing_ids = [id for id in success_case_ids_1 if id not in gitlab_ids]

    # get the success_case_ids that are in gitlab_ids
    ids = [id for id in success_case_ids_1 if id in gitlab_ids]

    print(ids)
    print(len(ids))

filter_results()