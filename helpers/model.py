from collections import Counter

action_event_list = []
eye_event_list = []


def classify_main_batch(action_batch):
    return Counter(action_batch).most_common(1)[0][0]


def classify_eye_batch(eye_batch):
    return Counter(eye_batch).most_common(1)[0][0]


def add_event(event, isActionCam):
    global action_event_list, eye_event_list
    previousEvent = action_event_list[-1] if isActionCam else eye_event_list[-1]
    if (
        previousEvent["label"] == event["label"]
        and event["frameStart"] - previousEvent["frameEnd"] < 15
    ):
        if isActionCam:
            action_event_list[-1]["frameEnd"] = event["frameEnd"]
        else:
            eye_event_list[-1]["frameEnd"] = event["frameEnd"]
    else:
        if isActionCam:
            action_event_list.append(event)
        else:
            eye_event_list.append(event)

    return


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
