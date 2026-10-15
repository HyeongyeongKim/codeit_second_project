"""Keep the HWP5 hyperlink reader moving past malformed control fields."""

from __future__ import annotations

import struct


def apply_hwp_parser_fix() -> None:
    """Patch only this Python process before HWP/HWPX documents are loaded."""
    from hwp_hwpx_parser.hwp5 import CTRL_ID_HYPERLINK, HWP5Reader

    def _extract_hyperlink_texts_from_para(self, para_data: bytes) -> list[str]:
        hyperlink_texts: list[str] = []
        i = 0

        while i < len(para_data) - 1:
            code = struct.unpack_from("<H", para_data, i)[0]

            if code == 0x03:
                if i + 6 > len(para_data):
                    i += 2
                    continue
                ctrl_id = struct.unpack_from("<I", para_data, i + 2)[0]

                if ctrl_id == CTRL_ID_HYPERLINK:
                    text_chars: list[str] = []
                    j = i + 14

                    while j < len(para_data) - 1:
                        c = struct.unpack_from("<H", para_data, j)[0]
                        if c == 0x04:
                            break
                        if c == 0x03:
                            j += 2
                        elif 0x20 <= c < 0x10000:
                            text_chars.append(chr(c))
                            j += 2
                        else:
                            j += 2

                    if text_chars:
                        hyperlink_texts.append("".join(text_chars))
                    i = j
                    continue
                i += 14
                continue
            if code == 0x04:
                i += 10
            elif code < 32:
                if code in (11, 12):
                    i += 10
                elif 15 <= code <= 23:
                    i += 14
                else:
                    i += 2
            else:
                i += 2

        return hyperlink_texts

    HWP5Reader._extract_hyperlink_texts_from_para = _extract_hyperlink_texts_from_para
