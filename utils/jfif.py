def as_short(twobytes):
    return ord(twobytes[0])*256 + ord(twobytes[1]);

def trim_at_null(str):
    index = str.find("\x00")
    if index == -1:
        return str
    return str[0:index]

def comments(file):
    with open(file, "rb") as jpg:
        start = jpg.read(4)
        if start != "\xff\xd8\xff\xe0":
            return None

        length = as_short(jpg.read(2))-2
        jpg.read(length)               # Skip the initial, non-comment, block

        comments = [];
        next = jpg.read(2);
        while next == "\xff\xfe":   # Comment block
            length = as_short(jpg.read(2))-2
            comments += [trim_at_null(jpg.read(length))]
            next = jpg.read(2);

    # creates dictionary from jfif tags, makes empty dict element if tag is listed but empty
    com_dict = dict([x.split(': ',1) if len(x.split(': ',1)) == 2 else x.split(': ',1)+[''] for x in comments])
    return com_dict

def gps(file):
    comms = comments(file)
    print file
    print comms
    try:
        lat = float(comms['latitude'])
        lon = float(comms['longitude'])
    except KeyError, e:
        lat = -1
        lon = -1
    return (lat, lon)
