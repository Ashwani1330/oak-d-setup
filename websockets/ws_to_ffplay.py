#!/usr/bin/env python3
import asyncio
import struct
import subprocess
import websockets

WS_SUB_URL = "ws://192.168.1.182:8765/sub"

async def main():
    # ffplay reads raw H264 from stdin
    ffplay = subprocess.Popen(
        ["ffplay", "-loglevel", "warning", "-fflags", "nobuffer", "-flags", "low_delay",
         "-framedrop", "-probesize", "32", "-analyzeduration", "0",
         "-f", "h264", "-i", "pipe:0"],
        stdin=subprocess.PIPE
    )

    try:
        async with websockets.connect(WS_SUB_URL, max_size=None) as ws:
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    if message.startswith(b"H264 "):
                        print("[viewer] stream header:", message.decode("ascii", "ignore").strip())
                        continue

                    if len(message) < 4:
                        continue
                    n = struct.unpack(">I", message[:4])[0]
                    payload = message[4:4+n]
                    if len(payload) != n:
                        continue

                    try:
                        ffplay.stdin.write(payload)
                        ffplay.stdin.flush()
                    except BrokenPipeError:
                        break
    finally:
        try:
            ffplay.stdin.close()
        except Exception:
            pass
        ffplay.terminate()

if __name__ == "__main__":
    asyncio.run(main())
