# Explore JPEG with Python

Explore JPEG standard with Python!

The first goal is to have a working decoder for baseline JPEG.

Resources:

- [Overview of JPEG](https://jpeg.org/jpeg/index.html)
- [JPEG Part 1 - ITU T.81](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)
- [JPEG Part 5 - JFIF](https://www.w3.org/Graphics/JPEG/jfif.pdf)

Process and Lessons I have learned:

- Implement a marker parser first. Look at ITU T.81 p.36 for the list of all markers. Look for `0xFF`, and the next byte determines the marker type. ITU T.81 Annex B will define the bitstream for each marker. Use JFIF (JPEG Part 5) to parse marker `APP0`, even though it is not important.
- For all markers, except for standalone markers, the first byte (after marker bytes) defines segment length. The standalone markers have no payload. They are `RSTx` (restart markers), `SOI` (start of image), `EOI` (end of image), and `TEM` (temporary).
- For a typical Baseline JPEG file, you only need to parse (in order of appearance):
  - `SOI`: start of image
  - `APP0`: JFIF (not important)
  - `DQT`: define quantization table (2 tables, 1 for Y - luminance, 1 for CbCr - chrominance)
  - `SOF0`: start of frame, baseline JPEG
  - `DHT`: define Huffman table (4 tables, 1 for each (DC, AC) x (luminance, chrominance))
  - `SOS`: start of scan
  - `EOI`: end of image
- Implement scan decoder. Refer to ITU T.81 F.2.2 p.104. This is the hardest part.
  - Scan length is not encoded, so we have to to decode the Huffman bitstream to know when the scan ends.
  - Special care is required when seeing `0xFF` in the Huffman bitstream, which involves **byte stuffing**. Refer to ITU T.81 F.2.2.5 p.110
  - Number of blocks is not encoded, but we can derive from image (frame) dimensions. Assume a typical 4:2:2 JPEG Baseline JPEG file, `n_blocks = ceil(width / 8) * ceil(height / 8) * 1.5`. We will decode until we obtain enough blocks.
- Encoded Huffman bitstream structure:
  - Sequence of MCUs (minimum-coded units), which contain **interleaved** 8x8 blocks from all components. E.g. Y1 Y2 Y3 Y4 Cb1 Cr1. Refer to ITU T.81 A.2 p.25 (NOTE: it is interleaved only if `SOS` defines more than 1 components, which should be true most of the time). Use the corresponding quantization tables and Huffman tables for each component.
  - Each block contains 1 DC coefficient and 63 AC coefficients in that order. Use the corresponding Huffman table for DC and AC.
- To decode each block (Refer to F.2):
  - Decode 1 DC coefficient
  - Decode 63 AC coefficients
  - Decode zig-zag encoding (can be done after dequantization)
  - Dequantize
  - Inverse DCT
  - Level shift (+128 for 8-bit precision)
- To combine blocks in an MCU:
  - Scale and place each block appropriately. E.g. 16x16 MCU
    - For Y component (4 blocks), place each block at each corner.
    - For Cb and Cr components (1 block each), scale each block up (from 8x8 to 16x16).
  - Convert YCbCr to RGB. Refer to JFIF p.3
- For each DCT coefficient, only the **size** (number of bits) is Huffman-encoded, the value is not.
