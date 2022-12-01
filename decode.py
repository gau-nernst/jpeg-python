import itertools
import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HuffmanTable:
    huffsize: Optional[tuple] = None
    huffcode: Optional[tuple] = None
    huffval: Optional[tuple] = None
    mincode: Optional[tuple] = None
    maxcode: Optional[tuple] = None
    valptr: Optional[tuple] = None


QuantizationTable = Tuple


@dataclass
class Component:
    x_sampling: int
    y_sampling: int
    q_table_id: int
    h_table_dc_id: int = 0  # to be updated by SOS
    h_table_ac_id: int = 0


@dataclass
class JPEGDecoderState:
    width: Optional[int] = None
    height: Optional[int] = None
    components: Optional[List[Component]] = None
    component_ids: Optional[List[int]] = None
    q_tables: List[Optional[QuantizationTable]] = field(default_factory=lambda: [None] * 4)
    huffman_tables_dc: List[Optional[HuffmanTable]] = field(default_factory=lambda: [None] * 4)
    huffman_tables_ac: List[Optional[HuffmanTable]] = field(default_factory=lambda: [None] * 4)


# itu-t81 p.36
MARKER_PREFIX = 0xFF
STANDALONE_MARKERS = tuple(f"RST{i}" for i in range(8)) + ("SOI", "EOI", "TEM")

MARKER_TO_BYTE = dict(
    TEM=0x01,  # temporary private use in arithmetic coding
    COM=0xFE,  # comment
)

for i in range(16):
    MARKER_TO_BYTE[f"SOF{i}"] = 0xC0 + i  # start of frame

MARKER_TO_BYTE.update(
    DHT=0xC4,  # define Huffman table
    JPG=0xC8,  # reserved
    DAC=0xCC,  # define arithmetic coding conditioning(s)
)

# restart interval termination
for i in range(8):
    MARKER_TO_BYTE[f"RST{i}"] = 0xD0 + i

MARKER_TO_BYTE.update(
    SOI=0xD8,  # start of image
    EOI=0xD9,  # end of image
    SOS=0xDA,  # start of scan
    DQT=0xDB,  # define quantization table
    DNL=0xDC,  # define number of lines
    DRI=0xDD,  # define restart interval
    DHP=0xDE,  # define hierarchical progression
    EXP=0xDF,  # expand reference component(s)
)

for i in range(16):
    MARKER_TO_BYTE[f"APP{i}"] = 0xE0 + i  # app segments

for i in range(14):
    MARKER_TO_BYTE[f"JPG{i}"] = 0xF0 + i  # reserved


BYTE_TO_MARKER = {v: k for k, v in MARKER_TO_BYTE.items()}


# a(u,v) * cos((i+1/2)*u*pi/N)
DCT_MATRIX: np.ndarray = (2 / 8) ** 0.5 * np.cos((np.arange(8) + 0.5)[None, :] * np.arange(8)[:, None] * np.pi / 8)
DCT_MATRIX[0, :] = (1 / 8) ** 0.5


def idct(x: np.ndarray):
    assert x.ndim == 2
    return DCT_MATRIX.T @ x @ DCT_MATRIX


# JFIF, p.3
YCbCr_offset = np.array([0.0, -128.0, -128.0]).reshape(1, 1, -1)
YCbCr_to_RGB = np.array(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.34414, -0.71414],
        [1.0, 1.772, 0.0],
    ]
)


def ycbcr_to_rgb(img: np.ndarray):
    assert img.ndim == 3 and img.shape[2] == 3, img.shape
    return (img + YCbCr_offset) @ YCbCr_to_RGB.T


def decode_zigzag(x: np.ndarray):
    assert len(x) == 64
    out = np.empty((8, 8), dtype=int)
    idx = i = j = 0
    while idx < len(x):
        out[i][j] = x[idx]
        idx += 1
        if i in (0, 7) and j % 2 == 0:
            j += 1
        elif j in (0, 7) and i % 2 == 1:
            i += 1
        elif (i + j) % 2 == 0:
            i, j = i - 1, j + 1
        else:
            i, j = i + 1, j - 1
    return out


def split_half_bytes(x: int):
    return x >> 4, x & 0x0F


def decode_marker(codes: bytes):
    assert codes[0] == MARKER_PREFIX
    assert codes[1] not in (0x00, 0xFF)
    return "RES" if 0x02 <= codes[1] <= 0xBF else BYTE_TO_MARKER[codes[1]]


HANDLER_TABLE = dict()


def register_handler(name):
    assert name not in HANDLER_TABLE

    def _register(handler):
        HANDLER_TABLE[name] = handler
        return handler

    return _register


@register_handler("APP0")
def handle_app0(payload: bytes, state: JPEGDecoderState):
    # https://www.w3.org/Graphics/JPEG/jfif.pdf
    if payload[:5] == b"JFIF\x00":
        version = f"{payload[5]}.{payload[6]:02d}"
        print(f"  version: {version}")

        units = ["no units", "dpi", "dpcm"][payload[7]]
        print(f"  units: {units}")

        x_density, y_density = struct.unpack(">HH", payload[8:12])
        print(f"  {x_density = }, {y_density = }")

        x_thumbnail, y_thumbnail = payload[12:14]
        print(f"  {x_thumbnail = }, {y_thumbnail = }")
        n = x_thumbnail * y_thumbnail
        assert len(payload) == 14 + n * 3
        if n > 0:
            return np.array(payload[14:]).reshape(y_thumbnail, x_thumbnail, 3)

    elif payload[:5] == b"JFXX\x00":
        # JFIF extension
        print(f"  JFIF extension (JFXX) is not supported")

    else:
        print("  unrecognized APP0 marker")


@register_handler("DQT")
def handle_dqt(payload: bytes, state: JPEGDecoderState):
    # itu-t81 B.2.4.1 p.39
    # DQT segment can contain multiple quantization tables
    pointer = 0
    while pointer < len(payload):
        precision, q_table_id = split_half_bytes(payload[pointer])
        print(f"  {q_table_id = }")

        table_size = 64 * (precision + 1)
        fmt = ">" + ("H" if precision else "B") * table_size
        q_table = struct.unpack(fmt, payload[pointer + 1 : pointer + 1 + table_size])
        state.q_tables[q_table_id] = q_table
        print(decode_zigzag(q_table))
        pointer += 1 + table_size


@register_handler("DRI")
def handle_dri(payload: bytes, state: JPEGDecoderState):
    (restart_interval,) = struct.unpack(">H", payload)
    print(f"  {restart_interval = }")


@register_handler("COM")
def handle_com(payload: bytes, state: JPEGDecoderState):
    print(f"  {payload = }")


@register_handler("SOF0")
def handle_sof0(payload: bytes, state: JPEGDecoderState):
    # itu-t81 B.2.2 p.35
    precision = payload[0]
    print(f"  {precision = }")

    height, width = struct.unpack(">HH", payload[1:5])
    print(f"  {width = }, {height = }")
    state.width = width
    state.height = height

    n_components = payload[5]
    print(f"  {n_components = }")

    state.components = [None] * n_components
    offset = 6
    for _ in range(n_components):
        component_id, sampling_rate, q_table_id = payload[offset : offset + 3]
        assert component_id <= n_components
        x_sampling, y_sampling = split_half_bytes(sampling_rate)
        print(f"    {component_id = }: {x_sampling = } {y_sampling = } {q_table_id = }")

        state.components[component_id - 1] = Component(x_sampling, y_sampling, q_table_id)
        offset += 3


@register_handler("DHT")
def handle_dht(payload: bytes, state: JPEGDecoderState):
    # itu-t81 B.2.4.2 p.40
    # DHT segment can contain multiple Huffman tables
    pointer = 0
    while pointer < len(payload):
        is_ac, h_table_id = split_half_bytes(payload[pointer])
        print(f"  {is_ac = }")
        print(f"  {h_table_id = }")

        bits = tuple(payload[pointer + 1 : pointer + 17])
        n_values = sum(bits)
        print(f"  {bits = }")
        print(f"  total number of codes = {sum(bits)}")

        huffval = tuple(payload[pointer + 17 : pointer + 17 + n_values])
        print(f"  {huffval = }")

        # itu-t81 p.50, Annex C
        # itu-t81 F.2.2.3 p.107
        huffsize, huffcode = [], []
        mincode, maxcode, valptr = [], [], []
        code, offset = 0, pointer
        for i, num_codes in enumerate(bits):
            mincode.append(code)
            valptr.append(len(huffcode))
            for _ in range(num_codes):
                huffsize.append(i + 1)
                huffcode.append(code)
                code, offset = code + 1, offset + 1
            maxcode.append(code - 1 if num_codes > 0 else -1)
            code = code << 1
        
        h_table = HuffmanTable(huffsize, huffcode, huffval, mincode, maxcode, valptr)
        (state.huffman_tables_ac if is_ac else state.huffman_tables_dc)[h_table_id] = h_table

        pointer += 17 + n_values


@register_handler("SOS")
def handle_sos(payload: bytes, state: JPEGDecoderState):
    # itu-t81 B.2.3 p.37
    n_components = payload[0]
    print(f"  {n_components = }")

    offset = 1
    state.component_ids = []
    for _ in range(n_components):
        component_id, h_table = payload[offset : offset + 2]
        h_table_dc_id, h_table_ac_id = split_half_bytes(h_table)
        print(f"  {component_id = } {h_table_dc_id = } {h_table_ac_id = }")

        state.component_ids.append(component_id - 1)
        state.components[component_id - 1].h_table_dc_id = h_table_dc_id
        state.components[component_id - 1].h_table_ac_id = h_table_ac_id
        offset += 2

    # not used for now
    start_of_select, end_of_select, sar = payload[offset:]
    print(f"  {start_of_select = }")
    print(f"  {end_of_select = }")
    print(f"  {sar = }")


def next_bit(f):
    # p.111
    while True:
        byte = f.read(1)[0]
        if byte == MARKER_PREFIX:
            byte2 = f.read(1)[0]
            if byte2 != 0:  # byte stuffing
                assert byte2 == MARKER_TO_BYTE["DNL"], hex(byte2)
                raise RuntimeError("DNL marker is not handled")

        for i in range(7, -1, -1):
            yield (byte >> i) & 1


def decode(bit_generator, h_table: HuffmanTable):
    # itu-t81, Figure F.16, DECODE
    size, code = 1, next(bit_generator)
    while code > h_table.maxcode[size - 1]:
        size += 1
        code = (code << 1) + next(bit_generator)
    j = h_table.valptr[size - 1] + code - h_table.mincode[size - 1]
    return h_table.huffval[j]


def receive(bit_generator, n_bits: int):
    # F.2.2.4, p.110, Figure F.17
    value = 0
    for _ in range(n_bits):
        value = (value << 1) + next(bit_generator)
    return value


def extend(value: int, n_bits: int):
    # p.105, Figure F.12
    return value + (-1 << n_bits) + 1 if n_bits > 0 and value < (1 << (n_bits - 1)) else value


def read_scan(f, state: JPEGDecoderState):
    n_components = len(state.components)
    max_x_sampling = max(state.components[idx].x_sampling for idx in state.component_ids)
    max_y_sampling = max(state.components[idx].y_sampling for idx in state.component_ids)

    x_blocks = math.ceil(state.width / 8)
    y_blocks = math.ceil(state.height / 8)

    x_scales = [max_x_sampling // state.components[idx].x_sampling for idx in range(n_components)]
    y_scales = [max_y_sampling // state.components[idx].y_sampling for idx in range(n_components)]

    image = np.empty((state.height, state.width, n_components), dtype=np.uint8)

    dc_coefs = [0] * n_components
    bit_generator = next_bit(f)
    for (mcu_y, mcu_x) in itertools.product(range(y_blocks // max_y_sampling), range(x_blocks // max_x_sampling)):
        mcu = np.empty((8 * max_y_sampling, 8 * max_x_sampling, n_components), dtype=np.uint8)
        for component_id in state.component_ids:
            component = state.components[component_id]
            h_dc_table = state.huffman_tables_dc[component.h_table_dc_id]
            h_ac_table = state.huffman_tables_ac[component.h_table_dc_id]
            q_table = state.q_tables[component.q_table_id]

            for (yi, xi) in itertools.product(range(component.y_sampling), range(component.x_sampling)):
                dct_coefs = [0] * 64

                # decode dc coefficient: itu-t81, p.104, F.2.2.1
                n_bits = decode(bit_generator, h_dc_table)
                diff = receive(bit_generator, n_bits)
                diff = extend(diff, n_bits)
                dc_coefs[component_id] += diff
                dct_coefs[0] = dc_coefs[component_id]

                # decode ac coefficients: F.2.2.2, p.105, Figure F.13
                # to understand, also read F.1.2.2 encode ac coefficients
                dct_idx = 1
                while dct_idx < 64:
                    composite = decode(bit_generator, h_ac_table)
                    if composite == 0xF0:   # ZRL - zero run-length
                        dct_idx += 16
                    elif composite == 0:    # EOB - end of block
                        break
                    else:
                        n_skip, n_bits = split_half_bytes(composite)
                        dct_idx += n_skip
                        value = receive(bit_generator, n_bits)
                        value = extend(value, n_bits)
                        dct_coefs[dct_idx] = value
                        dct_idx += 1

                dequantized = [v * q for v, q in zip(dct_coefs, q_table)]
                dct_block = decode_zigzag(dequantized)
                block = idct(dct_block)
                block += 128  # level shift

                block = np.repeat(block, x_scales[component_id], axis=1)
                block = np.repeat(block, y_scales[component_id], axis=0)
                x_block_size = 8 * x_scales[component_id]
                y_block_size = 8 * y_scales[component_id]
                mcu[
                    yi * y_block_size : (yi + 1) * y_block_size,
                    xi * x_block_size : (xi + 1) * x_block_size,
                    component_id,
                ] = block

        mcu = ycbcr_to_rgb(mcu)
        image[mcu_y * 16 : (mcu_y + 1) * 16, mcu_x * 16 : (mcu_x + 1) * 16, :] = mcu.round()

    return image


def decode_jpeg(f):
    state = JPEGDecoderState()
    marker = decode_marker(f.read(2))

    while True:
        print(marker)
        if marker == "EOI":
            break

        if marker in STANDALONE_MARKERS:
            marker = decode_marker(f.read(2))
            continue

        (length,) = struct.unpack(">H", f.read(2))
        print(f"  {length = }")
        payload = f.read(length - 2)

        HANDLER_TABLE[marker](payload, state)
        print()

        if marker == "SOS":
            image = read_scan(f, state)

        marker = decode_marker(f.read(2))

    return image


def main():
    import argparse

    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    image = decode_jpeg(open(args.path, "rb"))
    pil_image = np.array(Image.open(args.path))

    print(((image - pil_image) ** 2).mean() ** 0.5)


if __name__ == "__main__":
    main()
