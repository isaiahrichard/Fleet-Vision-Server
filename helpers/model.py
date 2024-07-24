from collections import Counter

action_event_list = []
eye_event_list = []


def classify_main_batch(action_batch):
    return Counter(action_batch).most_common(1)[0][0]


def classify_eye_batch(eye_batch):
    return Counter(eye_batch).most_common(1)[0][0]


# def add_eye_event(event):
#     global eye_event_list

#     if len(eye_event_list):
#         prev_event = eye_event_list[-1]
#         if (
#             prev_event["label"] == event["label"]
#             and event["frameStart"] - prev_event["frameEnd"] < 15
#         ):
#             eye_event_list[-1]["frameEnd"] = event["frameEnd"]
#             return

#     eye_event_list.append(event)


# def add_action_event(event):
#     global action_event_list

#     if len(action_event_list):
#         prev_event = action_event_list[-1]
#         if (
#             prev_event["label"] == event["label"]
#             and event["frameStart"] - prev_event["frameEnd"] < 15
#         ):
#             action_event_list[-1]["frameEnd"] = event["frameEnd"]
#             return

#     action_event_list.append(event)


# Potentially for future use
# main_actions_dict = {
#         "change_gear": 0,
#         "drinking": 0,
#         "hair_and_makeup": 0,
#         "phonecall_left": 0,
#         "phonecall_right": 0,
#         "radio": 0,
#         "reach_backseat": 0,
#         "reach_side": 0,
#         "safe_drive": 0,
#         "standstill_or_waiting": 0,
#         "talking_to_passenger": 0,
#         "texting_left": 0,
#         "texting_right": 0,
#         "unclassified": 0,
#         "yawning_with_hand": 0,
#         "yawning_without_hand": 0,
#     }
# eye_actions_dict = {
#         "open": 0,
#         "opening": 0,
#         "closed": 0,
#         "closing": 0,
#         "unclassified": 0,
#     }
