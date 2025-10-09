import re
import json

def get_boxed_contents(text: str):
    """~T~[~^~V~G~\__~I~@~\~I \boxed{...} ~Z~D~F~E__~H~T~L~A~L~W~J~K~O~N~M~L~I"""
    boxes = []
    i = 0
    needle = r'\boxed{'
    nlen = len(needle)

    while True:
        start = text.find(needle, i)
        if start == -1:
            break  # __~\~I~[~Z \boxed{
        j = start + nlen
        depth = 1  # ~[~E__~@__ {

        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == '\\':
                # __~G__~I~W__~H~B \{ ~H~V \}~I~L~A~E~M~J~J~C__~S~\~H~F~D~K~O__~U
                j += 2
                continue
            if ch == '{':
                depth += 1
                j += 1
                continue
            if ch == '}':
                depth -= 1
                if depth == 0:
                    # ~M~U~N~H~N__~K~J~K~O~L~E~M~Z~D~S~]~_~M__
                    boxes.append(text[start + nlen : j])
                    j += 1
                    break
                j += 1
                continue
            j += 1

        i = j  # ____~P~Q~P~N~_~I

    return boxes

def get_last_boxed(text: str):
    """~T~[~^~\~@~P~N~@__ \boxed{...} ~Z~D~F~E__~[~K~W| ~H~Y~T~[~^ None"""
    try:
        boxes = get_boxed_contents(text)
        return boxes[-1] if boxes else 'none'
    except:
        return None

def get_vote_result(text):
    tmp = get_last_boxed(text)
    try:
        tmp_json = json.loads(f"{{{tmp}}}")
    except:
        tmp_json = {}
    if tmp_json == {}:
        try:
            tmp_trans = tmp.replace('\\', '\\\\')
            tmp_json = json.loads(f"{{{tmp_trans}}}")
        except:
            tmp_json = {}
    if tmp_json == {}:
        try:
            tmp_json =  json.loads('{' + tmp.strip().strip('\\{}\n ') + '}')
        except:
            tmp_json = {}
    if tmp_json == {}:
        try:
            tmp_json =  json.loads('{' + tmp.strip().strip('\\{}\n ').replace('\\', '\\\\') + '}')
        except:
            tmp_json = {}
    tmp_json = dict(
        sorted(
            tmp_json.items(),
            key=lambda x: (isinstance(x[1], str), -x[1] if isinstance(x[1], (int, float)) else 0)
        )
    )
    return tmp_json
