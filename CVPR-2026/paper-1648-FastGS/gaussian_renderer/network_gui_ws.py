#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

host = "127.0.0.1"
port = 6009

import asyncio
import websockets
import threading
import struct

curr_id = -1
latest_width = 0
latest_height = 0
latest_result = bytes([])

async def echo(websocket, path):
    global curr_id
    global latest_result
    try:
        async for message in websocket:

            # If you expect binary data, you can check if message is an instance of bytes
            if isinstance(message, bytes):
                value = int.from_bytes(message, byteorder='big', signed=True)
                curr_id = value

                header = struct.pack('ii', latest_width, latest_height)  # Pack the two integers (height and width)

                # Send the entire tensor as one WebSocket message
                await websocket.send(header + latest_result)

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")

async def wuppi(host, port):
    async with websockets.serve(echo, host, port):
        await asyncio.Future()  # run forever

def run_asyncio_loop(wish_host, wish_port):
    asyncio.run(wuppi(wish_host, wish_port))

def init(wish_host, wish_port):
    thread = threading.Thread(target=run_asyncio_loop,args=[wish_host, wish_port])
    thread.start()
