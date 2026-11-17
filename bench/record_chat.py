#!/usr/bin/env python3
"""Record real terminal chat in a PTY, then render its timed output as a GIF.

Capture uses the standard library. Rendering needs Pillow and DejaVu Sans Mono.
The JSON retains every terminal output event, including input echo and waits.
"""
import argparse
import codecs
import errno
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import termios
import time

from harness import machine_state, read_json, sha256, write_json
from speed import source_hash


def record(args):
    binary, model = args.binary.resolve(), args.model.resolve()
    command = [str(binary), 'chat', '-m', str(model), '--threads', '4',
               '--isa', 'avx512', '--ctx-size', '256', '-n', '16']
    before = machine_state()
    model_hash, binary_hash = sha256(model), sha256(binary)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 100, 0, 0))
    started = time.monotonic()
    process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    decoder = codecs.getincrementaldecoder('utf-8')()
    events, transcript = [], ''
    inputs = ['My name is Ada. Reply with OK.',
              'What is my name? Reply with only the name.', '/reset', '/exit']
    sent = 0
    try:
        deadline = started + 180
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    data = os.read(master, 4096)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
                if not data:
                    break
                text = decoder.decode(data)
                events.append([round(time.monotonic() - started, 6), text])
                transcript += text
                if transcript.endswith('you> ') and sent < len(inputs):
                    time.sleep(0.6)
                    os.write(master, (inputs[sent] + '\n').encode())
                    sent += 1
            elif process.poll() is not None:
                break
        else:
            raise TimeoutError('chat recording exceeded its deadline')
        code = process.wait(timeout=10)
        if code or sent != len(inputs) or 'History cleared.' not in transcript:
            raise RuntimeError('chat did not complete the recorded conversation')
        write_json(args.output, dict(command=command, model=dict(file=model.name, sha256=model_hash),
            binary_sha256=binary_hash, source_sha256=source_hash(),
            parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            before=before, after=machine_state(), events=events, inputs=inputs,
            duration_seconds=round(time.monotonic() - started, 6), exit_code=code,
            transcript=transcript, notes=args.notes,
            rendering=dict(speed=1, frame_ms=100, width=960, height=340,
                           font='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', font_size=17)))
    finally:
        os.close(master)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def render(args):
    from PIL import Image, ImageDraw, ImageFont
    data = read_json(args.input)
    settings = data['rendering']
    font = ImageFont.truetype(settings['font'], settings['font_size'])
    width, height = settings['width'], settings['height']
    cell = font.getlength('M')
    columns = int((width - 32) / cell)
    lines, line, event = [], '', 0
    frames = []
    frame_ms = settings['frame_ms']
    # A fixed palette lets GIF store text changes as small rectangle updates.
    background = (22, 32, 43)
    colors = list(background)
    for foreground in [(240, 242, 245), (162, 217, 176), (169, 180, 192)]:
        for level in range(1, 6):
            colors.extend(round(a + (b - a) * level / 5) for a, b in zip(background, foreground))
    palette = Image.new('P', (1, 1))
    palette.putpalette(colors + [0] * (768 - len(colors)))
    for tick in range(int(data['duration_seconds'] * 1000 / frame_ms) + 2):
        seconds = tick * frame_ms / 1000
        while event < len(data['events']) and data['events'][event][0] <= seconds:
            for char in data['events'][event][1]:
                if char == '\r':
                    continue  # This CLI emits CRLF; it does not redraw lines.
                if char == '\n':
                    lines.append(line)
                    line = ''
                else:
                    if ord(char) < 32:
                        raise ValueError('unsupported terminal control character')
                    if len(line) == columns:
                        lines.append(line)
                        line = ''
                    line += char
            event += 1
        frame = Image.new('RGB', (width, height), '#16202b')
        draw = ImageDraw.Draw(frame)
        draw.text((16, 12), 'neuron chat | ' + data['model']['file'] + ' | Ghost', font=font, fill='#a2d9b0')
        draw.text((16, 38), 'AVX-512, 4 threads, context 256, reply limit 16', font=font, fill='#a9b4c0')
        for i, text in enumerate((lines + [line])[-10:]):
            draw.text((16, 77 + 22 * i), text, font=font, fill='#f0f2f5')
        # All elapsed time comes from the capture. No pauses are removed.
        draw.text((16, height - 28), f'{int(seconds)}s | real time, including load and prompt evaluation',
                  font=font, fill='#a9b4c0')
        frames.append(frame.quantize(palette=palette, dither=0))
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=frame_ms, loop=0, optimize=True)
    if args.output.stat().st_size > 1024 * 1024:
        raise ValueError('GIF exceeds the repository size guard')
    print(json.dumps(dict(file=str(args.output), bytes=args.output.stat().st_size,
                          frames=len(frames), duration_seconds=len(frames) * frame_ms / 1000)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    capture = sub.add_parser('record')
    capture.add_argument('--binary', type=Path, required=True)
    capture.add_argument('--model', type=Path, required=True)
    capture.add_argument('--notes', required=True)
    capture.add_argument('--output', type=Path, required=True)
    gif = sub.add_parser('render')
    gif.add_argument('--input', type=Path, required=True)
    gif.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'record':
        record(args)
    else:
        render(args)


if __name__ == '__main__':
    main()
