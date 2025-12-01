# MIT License
#
# Copyright (c) 2025 AMD-AGI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import re
import json
import time
import yaml
import subprocess

def until_servers_up(config):
    base_url = "http://localhost:{port}/v1/models"
    
    # with open(config_path) as f:
    #     config = yaml.safe_load(f)
        
    devices: list = config['gpu_index']
    tp: int = config['tp']
    
    partitions = [devices[i:i+tp] for i in range(0, len(devices), tp)]
    ports = [8040 + int(i[0]) for i in partitions]
    
    print(f"Waiting for servers on ports: {ports}")
    t0 = time.perf_counter()
    print_every = 60 * 1
    last_print = t0
    while True:
        ready_ports = []
        all_ready = True
        for port in ports:
            url = base_url.format(port=port)
            result = subprocess.run(['curl', '-s', url], capture_output=True)
            if result.returncode != 0:
                all_ready = False
                # print(f"Servers on port {port} not ready yet, dt: {(time.perf_counter() - t0)/60: .2f} min")
                # break
            else:
                ready_ports.append(port)
        if all_ready:
            print(f"All servers are up, dt: {(time.perf_counter() - t0) / 60: .2f} min")
            break
        curr = time.perf_counter()
        if curr - last_print > print_every:
            print(f"Ready ports: {len(ready_ports)}/{len(ports)}, dt: {int((curr - t0) / 60)} min")
            last_print = curr
        time.sleep(1)

def get_boxed_contents(text: str):
    r"""~T~[~^~V~G~\__~I~@~\~I \boxed{...} ~Z~D~F~E__~H~T~L~A~L~W~J~K~O~N~M~L~I"""
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
    r"""~T~[~^~\~@~P~N~@__ \boxed{...} ~Z~D~F~E__~[~K~W| ~H~Y~T~[~^ None"""
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
