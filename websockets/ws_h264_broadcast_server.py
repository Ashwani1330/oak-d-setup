#!/usr/bin/env python3
import asyncio
import struct
import websockets

HOST = "0.0.0.0"
PORT = 8765

subscribers = set()
latest_header = None  # optional text header sent by publisher


async def handler(ws):
    global latest_header

    path = ws.request.path  # "/pub" or "/sub"

    if path == "/sub":
        subscribers.add(ws)
        try:
            # Send the latest header to new subscribers (if we have it)
            if latest_header is not None:
                await ws.send(latest_header)
            await ws.wait_closed()
        finally:
            subscribers.discard(ws)
        return

    if path != "/pub":
        await ws.close(code=1008, reason="Use /pub or /sub")
        return

    print("[server] publisher connected")
    try:
        async for message in ws:
            # First message may be ASCII header like: b"H264 640x400 30\n"
            if isinstance(message, (bytes, bytearray)):
                if message.startswith(b"H264 "):
                    latest_header = bytes(message)
                    # Forward header to all current subscribers
                    dead = []
                    for s in subscribers:
                        try:
                            await s.send(latest_header)
                        except Exception:
                            dead.append(s)
                    for s in dead:
                        subscribers.discard(s)
                    continue

                # Otherwise it's framed: 4-byte big-endian length + data
                if len(message) < 4:
                    continue
                n = struct.unpack(">I", message[:4])[0]
                payload = message[4:4+n]
                if len(payload) != n:
                    continue

                # Broadcast to subscribers (best-effort)
                dead = []
                for s in subscribers:
                    try:
                        await s.send(message)  # keep same framing
                    except Exception:
                        dead.append(s)
                for s in dead:
                    subscribers.discard(s)
            else:
                # Ignore non-binary
                pass

    except websockets.ConnectionClosed:
        pass
    finally:
        print("[server] publisher disconnected")


async def main():
    async with websockets.serve(handler, HOST, PORT, max_size=None):
        print(f"[server] listening on ws://{HOST}:{PORT} (/pub, /sub)")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
