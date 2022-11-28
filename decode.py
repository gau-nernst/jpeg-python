import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HuffmanTable:
    huffsize: Optional[np.ndarray] = None
    huffcode: Optional[np.ndarray] = None
    huffval: Optional[np.ndarray] = None
    mincode: Optional[np.ndarray] = None
    maxcode: Optional[np.ndarray] = None
    valptr: Optional[np.ndarray] = None


@dataclass
class QuantizationTable:
    data: Optional[np.ndarray] = None


@dataclass
class JPEGDecoderState:
    huffman_tables_dc: Tuple[HuffmanTable] = tuple(HuffmanTable() for _ in range(4))
    huffman_tables_ac: Tuple[HuffmanTable] = tuple(HuffmanTable() for _ in range(4))
    q_tables: Tuple[QuantizationTable] = tuple(QuantizationTable() for _ in range(4))


# https://www.w3.org/Graphics/JPEG/itu-t81.pdf

# itu-t81 p.36
MARKER_PREFIX = 0xFF
STANDALONE_MARKERS = tuple(f"RST{i}" for i in range(8)) + ("SOI", "EOI", "TEM")

BYTE_TO_MARKER = {
    0x01: "TEM",  # temporary private use in arithmetic coding
    0xFE: "COM",  # comment
}

for i in range(16):
    BYTE_TO_MARKER[0xC0 + i] = f"SOF{i}"  # start of frame

BYTE_TO_MARKER.update(
    {
        0xC4: "DHT",  # define Huffman table
        0xC8: "JPG",  # reserved
        0xCC: "DAC",  # define arithmetic coding conditioning(s)
    }
)

# restart interval termination
for i in range(8):
    BYTE_TO_MARKER[0xD0 + i] = f"RST{i}"

BYTE_TO_MARKER.update(
    {
        0xD8: "SOI",  # start of image
        0xD9: "EOI",  # end of image
        0xDA: "SOS",  # start of scan
        0xDB: "DQT",  # define quantization table
        0xDC: "DNL",  # define number of lines
        0xDD: "DRI",  # define restart interval
        0xDE: "DHP",  # define hierarchical progression
        0xDF: "EXP",  # expand reference component(s)
    }
)

for i in range(16):
    BYTE_TO_MARKER[0xE0 + i] = f"APP{i}"  # app segments

for i in range(14):
    BYTE_TO_MARKER[0xF0 + i] = f"JPG{i}"  # reserved


def decode_zigzag(x: np.ndarray):
    assert len(x) == 64
    out = np.empty((8, 8), dtype=x.dtype)
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


def handle_dqt(payload: bytes, state: JPEGDecoderState):
    # itu-t81 B.2.4.1 p.39
    # DQT segment can contain multiple quantization tables
    pointer = 0
    while pointer < len(payload):
        precision, q_table_id = split_half_bytes(payload[pointer])
        dtype = np.dtype(">uint16") if precision else np.uint8
        print(f"  {dtype = }")
        print(f"  {q_table_id = }")

        table_size = 64 * (precision + 1)
        table = np.frombuffer(payload[pointer + 1 : pointer + 1 + table_size], dtype=dtype)
        table = decode_zigzag(table)
        state.q_tables[q_table_id].data = table
        print(table)
        pointer += 1 + table_size


def handle_dri(payload: bytes, state: JPEGDecoderState):
    restart_interval = struct.unpack(">H", payload)
    print(f"  {restart_interval = }")


def handle_com(payload: bytes, state: JPEGDecoderState):
    print(f"  {payload = }")


def handle_sof0(payload: bytes, state: JPEGDecoderState):
    bits = payload[0]
    print(f"  {bits} bits per pixel per component")

    height, width = struct.unpack(">HH", payload[1:5])
    print(f"  {width = }, {height = }")

    num_components = payload[5]
    print(f"  {num_components = }")

    offset = 6
    for _ in range(num_components):
        component_id, sampling_rate, q_table_id = payload[offset : offset + 3]
        sampling_rate = split_half_bytes(sampling_rate)
        print(f"    {component_id = }: {sampling_rate = } {q_table_id = }")
        offset += 3


def handle_dht(payload: bytes, state: JPEGDecoderState):
    # itu-t81 B.2.4.2 p.40
    # DHT segment can contain multiple Huffman tables
    pointer = 0
    while pointer < len(payload):
        is_ac, h_table_id = split_half_bytes(payload[pointer])
        _table = (state.huffman_tables_ac if is_ac else state.huffman_tables_dc)[h_table_id]
        print(f"  {is_ac = }")
        print(f"  {h_table_id = }")

        bits = tuple(payload[pointer + 1 : pointer + 17])
        n_values = sum(bits)
        print(f"  {bits = }")
        print(f"  total number of codes = {sum(bits)}")

        huffval = tuple(payload[pointer + 17 : pointer + 17 + n_values])
        _table.huffval = huffval
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
        _table.huffsize = tuple(huffsize)
        _table.huffcode = tuple(huffcode)
        _table.mincode = tuple(mincode)
        _table.maxcode = tuple(maxcode)
        _table.valptr = tuple(valptr)

        curr_size, curr_codes = 1, []
        for size, code in zip(huffsize, huffcode):
            if size > curr_size:
                fmt = f"{{:0{curr_size}b}}"
                print(f"  {curr_size:2d}-bit ({len(curr_codes)} codes):", *list(map(fmt.format, curr_codes)))
                curr_size, curr_codes = curr_size + 1, []
            curr_codes.append(code)
        fmt = f"{{:0{curr_size}b}}"
        print(f"  {curr_size:2d}-bit ({len(curr_codes)} codes):", *list(map(fmt.format, curr_codes)))

        pointer += 17 + n_values


def handle_sos(payload: bytes, state: JPEGDecoderState):
    # itu-t81 B.2.3 p.41
    num_components = payload[0]
    print(f"  {num_components = }")

    offset = 1
    for _ in range(num_components):
        component_id, h_table = payload[offset : offset + 2]
        h_table = split_half_bytes(h_table)
        print(f"  {component_id = } {h_table = }")
        offset += 2

    start_of_select, end_of_select, sar = payload[offset:]
    print(f"  {start_of_select = }")
    print(f"  {end_of_select = }")
    print(f"  {sar = }")


def read_scan(f, state: JPEGDecoderState):
    # read until seeing 0xFFxx, while skipping 0xFF00 and 0xFFD0-0xFFD7
    buffer = []

    h_table = state.huffman_tables_dc[0]
    size, code = 0, 0
    values = []
    while True:
        byte = f.read(1)[0]
        print(f"{byte:08b}")
        for i in range(7, -1, -1):
            size += 1
            code = (code << 1) + ((byte >> i) & 1)
            if code > h_table.maxcode[size - 1]:
                continue

            print("  size:", size)
            print("  code:", "{:b}".format(code))
            j = h_table.valptr[size - 1] + code - h_table.mincode[size - 1]
            values.append(h_table.huffval[j])

            if len(values) > 100:
                print(values)
                raise

            size, code = 0, 0

        # buffer.append(f.read(1)[0])
        # if buffer[-1] == MARKER_PREFIX:
        #     continue

        # buffer.append(f.read(1)[0])
        # if buffer[-1] == 0x00 or 0xD0 <= buffer[-1] <= 0xD7:
        #     continue

        # break

    return buffer[:-2], decode_marker(buffer[-2:])


HANDLER_TABLE = dict(
    APP0=handle_app0,
    DQT=handle_dqt,
    DHT=handle_dht,
    DRI=handle_dri,
    SOF0=handle_sof0,
    SOS=handle_sos,
    COM=handle_com,
)


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

        # https://www.ccoderun.ca/programming/2017-01-31_jpeg/
        # https://www.youtube.com/watch?v=TlrNCT15NM4

        HANDLER_TABLE[marker](payload, state)
        print()

        if marker == "SOS":
            image_data, marker = read_scan(f, state)
            continue

        marker = decode_marker(f.read(2))


if __name__ == "__main__":
    path = "../../ZooA_1581.jpg"
    f = open(path, "rb")
    decode_jpeg(f)
