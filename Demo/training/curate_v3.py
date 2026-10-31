"""Freeze the second visually reviewed expansion; never infer labels from a detector."""
import copy
import json
import shutil

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, export, write_json


# Proposal numbers refer to expansion/proposals.json and sheet_00..23.jpg.
# All 94 proposals were visually inspected. Entire ambiguous images stay pending.
PENDING = {
    17: 'Đầu người bị cắt gần hết ở mép trái; không suy lớp từ frame bên cạnh.',
    21: 'Người áo xanh sau ô tô chưa rõ đang đi bộ hay ngồi xe.',
    24: 'Đầu người bị cắt ở mép ảnh; giữ riêng để rà lại.',
    42: 'Thiếu người và xe ở mép trái, phần người bị cắt và nhiễu.',
    48: 'Thiếu nhãn giữa ba xe/người chồng lấp; ảnh xoay và nhiễu mạnh.',
    49: 'Hai người chồng lấp nhưng chỉ có nhãn xe; cần tách box và xác nhận mũ.',
    50: 'Người phía phải bị cột che, chưa rõ có ngồi xe.',
    52: 'Người/xe bị che phía sau quầy cần xác nhận, không bỏ thành nền.',
    53: 'Mũ người ngoài cùng phải chưa phân biệt rõ mũ bảo hiểm và mũ vải.',
    55: 'Thiếu nhãn người/xe bị cắt phía phải.',
    57: 'Hai người trên xe phía sau bị gộp/thiếu nhãn.',
    58: 'Box gộp hai người trên xe gần; người lái bị che nhiều.',
    59: 'Thiếu xe sau thanh chắn bên trái; cần rà toàn bộ phần bị che.',
    60: 'Thiếu xe bên trái và hình dạng mũ phía phải chưa rõ.',
    62: 'Thiếu nhãn xe đỗ ở mép phải; phần bị che cần rà lại.',
    79: 'Mũ người phía sau chưa phân biệt chắc chắn mũ vải/mũ bảo hiểm.',
    93: 'Thiếu xe bên trái, mũ người phía phải cần xác nhận.',
}
# Manually measured motorcycle envelopes in each 960x540 sheet tile, including mirrors.
REPLACE = {
    3: {0: [0, 398, 186, 488, 335]},
    4: {0: [0, 335, 179, 473, 430]},
    5: {0: [0, 276, 217, 423, 349]},
    10: {0: [0, 742, 177, 960, 538]},
    12: {0: [0, 534, 137, 619, 264]},
    13: {0: [0, 385, 175, 495, 270]},
    15: {0: [0, 447, 161, 663, 470]},
    19: {0: [0, 543, 184, 595, 303]},
    25: {0: [0, 650, 232, 955, 537]},
    26: {0: [0, 510, 161, 698, 405]},
    27: {0: [0, 604, 199, 732, 396]},
    28: {0: [0, 514, 135, 594, 252]},
    29: {0: [0, 519, 202, 576, 303]},
    31: {0: [0, 499, 205, 552, 310]},
    32: {0: [0, 465, 206, 512, 286]},
    36: {0: [0, 485, 163, 600, 383]},
    37: {0: [0, 505, 199, 572, 339]},
    38: {0: [0, 490, 206, 550, 313]},
    39: {0: [0, 496, 211, 552, 305]},
    41: {0: [0, 519, 218, 627, 404]},
    44: {0: [0, 109, 211, 301, 501]},
    45: {0: [0, 223, 223, 441, 443], 1: [0, 629, 213, 739, 383]},
    46: {0: [0, 455, 224, 587, 410]},
    47: {0: [0, 246, 201, 426, 474]},
    54: {0: [0, 512, 270, 641, 459]},
    56: {0: [0, 452, 295, 590, 494]},
    63: {0: [0, 568, 245, 706, 471], 1: [0, 732, 207, 884, 429]},
    64: {0: [0, 226, 228, 285, 319]},
    67: {0: [0, 348, 172, 450, 315]},
    69: {0: [0, 443, 153, 610, 353]},
    70: {0: [0, 326, 189, 417, 353]},
    73: {0: [0, 532, 214, 675, 424]},
    74: {0: [0, 493, 221, 581, 367]},
    78: {0: [0, 354, 217, 511, 467]},
    82: {0: [0, 288, 246, 385, 399]},
    84: {0: [0, 389, 123, 598, 329]},
    88: {0: [0, 427, 210, 506, 331]},
    90: {0: [0, 541, 145, 744, 453]},
    91: {0: [0, 453, 214, 537, 340]},
    92: {0: [0, 540, 227, 677, 439]},
}


def main():
    output = ROOT/'dataset_v3'
    if output.exists():
        raise FileExistsError('Dataset v3 is frozen; use another version.')
    rows = json.loads((ROOT/'inventory_v2.json').read_text(encoding='utf-8'))
    original = json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))
    reviews = json.loads((ROOT/'reviews_v2.json').read_text(encoding='utf-8'))
    proposals = json.loads((ROOT/'review_round3/expansion/proposals.json').read_text(encoding='utf-8'))
    assert len(proposals) == 94
    decisions = []
    for number, proposal in enumerate(proposals):
        row = original[proposal['index']]
        boxes = copy.deepcopy(proposal['labels'])
        reason = 'Rà trực quan toàn ảnh trên contact sheet mở rộng; kiểm tra từng người, mũ và xe đỗ.'
        if number in PENDING:
            decision = dict(status='pending', reason=PENDING[number])
        else:
            for index, pixels in REPLACE.get(number, {}).items():
                assert boxes[index][0] == 0
                boxes[index] = normalize_boxes([pixels], [960, 540])[0]
            if number in REPLACE:
                reason += ' Chỉnh box xe để bao phần gương nhìn thấy.'
            if number == 22:
                assert boxes[4][0] == 2
                del boxes[4]
                reason += ' Xóa nhãn người không đội mũ trên xe đỏ đỗ trống.'
            if number == 28:
                assert boxes[1][0] == 1
                boxes[1][0] = 2
                reason += ' Sửa lớp 1 thành 2: đầu trần, nhìn thấy tóc.'
            if len(boxes) > proposal['original_count']:
                reason += ' Bổ sung xe đỗ từ phép chiếu hình học đã kiểm tra bằng mắt; không dùng detector.'
            decision = dict(status='corrected' if boxes != row['labels'] else 'accepted',
                            labels=boxes, reason=reason, source_verified=True)
        decision['label_sha256'] = row['label_sha256']
        reviews[row['id']] = decision
        decisions.append(dict(proposal=number, id=row['id'], evidence=f'review_round3/expansion/sheet_{number//4:02d}.jpg', **decision))
    write_json(ROOT/'expansion_decisions_v3.json', decisions)
    write_json(ROOT/'reviews_v3.json', reviews)
    export(rows, reviews, output, inherit_manifest=ROOT/'dataset_v2/manifest.json')
    for name in ['attribution.json', 'ATTRIBUTION.md', 'external_candidates_excluded.json']:
        shutil.copy2(ROOT/'dataset_v2'/name, output/name)
    print((output/'statistics.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
