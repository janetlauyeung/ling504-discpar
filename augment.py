def augment(data):
    # If you want to engineer some table features
    # before creating a dev partition, this is the place!
    data["First-Queue-Len"] = data["First-Queue"].str.len()
    return data
