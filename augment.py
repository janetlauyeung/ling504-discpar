def any_word_in_common(x, col1, col2):
    in_common = []
    for i in range(len(x)):
        toks1 = x[col1, i].split(" ")
        toks2 = x[col2, i].split(" ")
        in_common.append(any(w1 == w2 for w1 in toks1 for w2 in toks2))
    return in_common

#First-Queue
#Top1-Stack
def augment(data):
    # If you want to engineer some table features
    # before creating a dev partition, this is the place!
    data["First-Queue-Len"] = data["First-Queue"].str.len()
    return data["StackQueueCommonWord"] = any_word_in_common(data, "First-Queue", "Top1-Stack")
