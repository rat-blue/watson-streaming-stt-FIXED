#!/usr/bin/env python
#
# Copyright 2016 IBM
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import argparse
import base64
import configparser
import json
import threading
import time

import pyaudio
import websocket
from websocket._abnf import ABNF

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
FINALS = []
LAST = None

# Updated REGION_MAP for fixed API url
REGION_MAP = {
    'us-east': 'api.us-east.speech-to-text.watson.cloud.ibm.com',
    'us-south': 'api.us-south.speech-to-text.watson.cloud.ibm.com',
    'eu-gb': 'api.eu-gb.speech-to-text.watson.cloud.ibm.com',
    'eu-de': 'api.eu-de.speech-to-text.watson.cloud.ibm.com',
    'au-syd': 'api.au-syd.speech-to-text.watson.cloud.ibm.com',
    'jp-tok': 'api.jp-tok.speech-to-text.watson.cloud.ibm.com',
}

def read_audio(ws, timeout):
    """Read audio and send it to the websocket port."""
    global RATE
    p = pyaudio.PyAudio()
    RATE = int(p.get_default_input_device_info()['defaultSampleRate'])
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    rec = timeout or RECORD_SECONDS

    for i in range(0, int(RATE / CHUNK * rec)):
        data = stream.read(CHUNK)
        ws.send(data, ABNF.OPCODE_BINARY)

    stream.stop_stream()
    stream.close()
    print("* done recording")

    data = {"action": "stop"}
    ws.send(json.dumps(data).encode('utf8'))
    time.sleep(1)
    ws.close()

    p.terminate()

def on_message(self, msg):
    """Print whatever messages come in."""
    global LAST
    data = json.loads(msg)
    if "results" in data:
        if data["results"][0]["final"]:
            FINALS.append(data)
            LAST = None
        else:
            LAST = data
        print(data['results'][0]['alternatives'][0]['transcript'])

def on_error(self, error):
    """Print any errors."""
    print(error)

def on_close(ws, close_status_code, close_msg):
    """Upon close, print the complete and final transcript."""
    global LAST
    if LAST:
        FINALS.append(LAST)
    transcript = "".join([x['results'][0]['alternatives'][0]['transcript']
                          for x in FINALS])
    print(transcript)
    print(f"Connection closed with status: {close_status_code}, message: {close_msg}")

def on_open(ws):
    """Triggered as soon a we have an active connection."""
    args = ws.args
    data = {
        "action": "start",
        "content-type": "audio/l16;rate=%d" % RATE,
        "continuous": True,
        "interim_results": True,
        "word_confidence": True,
        "timestamps": True,
        "max_alternatives": 3
    }

    ws.send(json.dumps(data).encode('utf8'))
    threading.Thread(target=read_audio, args=(ws, args.timeout)).start()

def get_url():
    config = configparser.RawConfigParser()
    config.read('speech.cfg')
    region = config.get('auth', 'region')
    host = REGION_MAP[region]
    return f"wss://{host}/v1/recognize?model=en-US_BroadbandModel"

def get_auth():
    config = configparser.RawConfigParser()
    config.read('speech.cfg')
    apikey = config.get('auth', 'apikey')
    return ("apikey", apikey)

def parse_args():
    parser = argparse.ArgumentParser(description='Transcribe Watson text in real time')
    parser.add_argument('-t', '--timeout', type=int, default=5)
    args = parser.parse_args()
    return args

def main():
    headers = {}
    userpass = ":".join(get_auth())
    headers["Authorization"] = "Basic " + base64.b64encode(userpass.encode()).decode()
    url = get_url()

    ws = websocket.WebSocketApp(url,
                                header=headers,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.args = parse_args()
    ws.run_forever()

if __name__ == "__main__":
    main()
